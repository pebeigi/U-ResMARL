"""Published model cores with explicit local loss/sampling adapters."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .data import ctrl_namespaces
from .vendor.encoder import Encoder
from .vendor.decoder import Decoder
from .vendor.ctg_arch import DiT


def masked_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


class CtRLSim(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        upstream = cfg.upstream()
        self.encoder, self.decoder = Encoder(upstream), Decoder(upstream)

    def forward(self, data):
        data = ctrl_namespaces(data)
        return self.decoder(data, self.encoder(data, eval=not self.training), eval=not self.training)

    def loss(self, data, action_mask, return_mask):
        out = self(data)
        targets = data['agent']
        # Actions are return-conditioned; censored returns cannot supervise them.
        mask = action_mask*return_mask
        action_ce = F.cross_entropy(out['action_preds'].reshape(-1, self.cfg.accel_bins*self.cfg.steer_bins),
                                    targets['actions'].reshape(-1), reduction='none').reshape_as(mask)
        action_loss = masked_mean(action_ce, mask)
        logits = out['rtg_preds'].reshape(*mask.shape, self.cfg.return_bins, 3)
        return_loss = sum(masked_mean(F.cross_entropy(logits[..., c].reshape(-1, self.cfg.return_bins),
            targets['rtgs'][..., c].reshape(-1), reduction='none').reshape_as(mask), mask) for c in range(3))
        states = targets['agent_states']
        prediction = out['state_preds'].reshape(*mask.shape, self.cfg.context, 2)
        numerator, denominator = states.new_zeros(()), states.new_zeros(())
        for t in range(self.cfg.context-1):
            count = self.cfg.context-t-1
            valid = states[:, :, t+1:, -1]*states[:, :, t, -1:]
            error = (prediction[:, :, t, :count]-states[:, :, t+1:, :2]).square().sum(-1)
            numerator = numerator+(error*valid).sum()
            denominator = denominator+valid.sum()
        state_loss = numerator/(200*denominator.clamp_min(1))
        return action_loss+return_loss+state_loss, dict(action=float(action_loss.detach()),
            returns=float(return_loss.detach()), future_state=float(state_loss.detach()))

    @torch.no_grad()
    def sample(self, data, generator):
        """Sample predicted factored returns, tilt, then sample conditioned actions."""
        out = self(data)
        logits = out['rtg_preds'][:, :, -1].reshape(-1, self.cfg.return_bins, 3)
        values = torch.linspace(-1, 1, self.cfg.return_bins, device=logits.device)
        tilts = (self.cfg.goal_tilt, self.cfg.vehicle_tilt, self.cfg.road_tilt)
        selected = torch.stack([torch.multinomial((logits[..., c]+tilts[c]*values).softmax(-1),
                                                 1, generator=generator).squeeze(-1) for c in range(3)], -1)
        data['agent']['rtgs'][:, :, -1] = selected.reshape(data['agent']['rtgs'].shape[0], -1, 3)
        action_logits = self(data)['action_preds'][:, :, -1]
        tokens = torch.multinomial(action_logits.reshape(-1, action_logits.shape[-1]).softmax(-1),
                                   1, generator=generator).reshape(action_logits.shape[:2])
        return tokens, selected


class CTGPlusPlus(nn.Module):
    """CtRL-Sim authors' CTG++ DiT, x0 DDPM, optional local cost guidance.

    Uses all consecutive reverse steps (no invalid skipping of DDPM posteriors).
    The released reimplementation samples at half noise scale; retained here.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.denoiser = DiT(cfg.upstream(diffusion=True))
        steps = torch.arange(cfg.diffusion_steps+1, dtype=torch.float64)
        cumulative = torch.cos(((steps/cfg.diffusion_steps+.008)/1.008)*math.pi/2).square()
        cumulative = cumulative/cumulative[0]
        beta = (1-cumulative[1:]/cumulative[:-1]).clamp(0, .999).float()
        alpha = 1-beta; abar = alpha.cumprod(0)
        prev = torch.cat([torch.ones(1), abar[:-1]])
        for name, values in dict(beta=beta, abar=abar,
            posterior_variance=beta*(1-prev)/(1-abar),
            coef1=beta*prev.sqrt()/(1-abar), coef2=(1-prev)*alpha.sqrt()/(1-abar)).items():
            self.register_buffer(name, values)

    def q_sample(self, target, t, noise):
        a = self.abar[t].reshape(-1, 1, 1, 1)
        return a.sqrt()*target+(1-a).sqrt()*noise

    def loss(self, cond, target, mask):
        t = torch.randint(self.cfg.diffusion_steps, (target.shape[0],), device=target.device)
        noisy = self.q_sample(target, t, torch.randn_like(target))
        prediction = self.denoiser(noisy, cond, t, eval=not self.training)
        weight = torch.ones_like(target)
        weight[:, :, 0, -2:] = 10.
        error = (prediction-target).square()*weight
        loss = masked_mean(error.mean(-1), mask)
        return loss, {'diffusion': float(loss.detach())}

    @torch.no_grad()
    def sample(self, cond, generator, guidance=None):
        shape = (cond[0].shape[0], cond[0].shape[1], self.cfg.horizon, 7)
        x = .5*torch.randn(shape, device=self.abar.device, generator=generator)
        for step in reversed(range(self.cfg.diffusion_steps)):
            t = torch.full((shape[0],), step, device=x.device, dtype=torch.long)
            clean = self.denoiser(x, cond, t, eval=True)
            mean = self.coef1[step]*clean+self.coef2[step]*x
            if guidance is not None and self.cfg.guidance_scale > 0:
                with torch.enable_grad():
                    proposal = mean.detach().requires_grad_(True)
                    cost = guidance(proposal)
                    grad, = torch.autograd.grad(cost, proposal)
                # Posterior covariance scaling, with bounded gradient norm for stability.
                norm = grad.flatten(1).norm(dim=1).clamp_min(1).view(-1, 1, 1, 1)
                mean = mean-self.cfg.guidance_scale*self.posterior_variance[step]*grad/norm
            noise = .5*torch.randn(shape, device=x.device, generator=generator) if step else 0.
            x = mean+self.posterior_variance[step].sqrt()*noise
        return x


def bicycle_rollout(initial, actions, cfg):
    """Differentiable copy of repository integration, for guidance only."""
    pos, heading = initial[..., :2], initial[..., 4]
    speed = initial[..., 2:4].norm(dim=-1)
    positions, headings = [], []
    for t in range(actions.shape[-2]):
        accel = actions[..., t, 0].clamp(-cfg.max_accel, cfg.max_accel)
        steering = actions[..., t, 1].clamp(-cfg.max_steer, cfg.max_steer)
        next_speed = (speed+accel*cfg.dt).clamp(0, cfg.max_speed)
        yaw_speed = speed
        if cfg.steer_from_rest:
            yaw_speed = torch.maximum(speed, next_speed).clamp_min(cfg.min_steer_speed)
        heading = heading+yaw_speed/cfg.wheelbase*steering.tan()*cfg.dt
        pos = pos+next_speed[..., None]*torch.stack([heading.cos(), heading.sin()], -1)*cfg.dt
        positions.append(pos); headings.append(heading); speed = next_speed
    return torch.stack(positions, -2), torch.stack(headings, -1)


def make_guidance(initial, goals, corridor, cfg):
    """Optional differentiable two-disc collision, road and route-goal costs.

    This is a documented local guidance adapter, not the original LLM/STL layer.
    Execution still uses the shared simulator's exact boundary/OBB filters.
    """
    device = initial.device
    polygon = getattr(corridor, 'roadway', None)
    if polygon is not None:
        # Bilinear signed-distance guidance respects holes and concave curbs.
        # This is differentiable guidance; exact footprint containment remains
        # the responsibility of the common execution filter.
        import numpy as np
        import shapely
        cache = getattr(corridor, '_guidance_grid', None)
        if cache is None:
            lo = np.array(polygon.bounds[:2])-5.
            hi = np.array(polygon.bounds[2:])+5.
            nx, ny = np.ceil(hi-lo).astype(int)+1
            x, y = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
            points = shapely.points(np.c_[x.ravel(), y.ravel()])
            distance = shapely.distance(points, polygon.boundary)
            signed = np.where(shapely.covers(polygon, points), distance, -distance).reshape(ny, nx)
            cache = (signed.astype(np.float32), lo, hi)
            corridor._guidance_grid = cache
        sdf = torch.as_tensor(cache[0], device=device)[None, None]
        lo, hi = (torch.as_tensor(v, device=device, dtype=torch.float32) for v in cache[1:])
    else:
        center = torch.as_tensor(corridor.center, dtype=torch.float32, device=device)
        lower = torch.as_tensor(corridor.lower, dtype=torch.float32, device=device)
        upper = torch.as_tensor(corridor.upper, dtype=torch.float32, device=device)
    active = initial[..., -1] > 0
    pair_mask = active[:, :, None] & active[:, None, :]
    pair_mask &= ~torch.eye(initial.shape[1], device=device, dtype=torch.bool)[None]

    def cost(sample):
        actions = sample[..., -2:]*sample.new_tensor([cfg.max_accel, cfg.max_steer])
        positions, headings = bicycle_rollout(initial, actions, cfg)
        direction = torch.stack([headings.cos(), headings.sin()], -1)
        offset = (cfg.length-cfg.width)/2
        discs = torch.stack([positions-offset*direction, positions+offset*direction], -2)
        delta = discs[:, :, None, :, :, None]-discs[:, None, :, :, None, :]
        distance = (delta.square().sum(-1)+1e-6).sqrt().amin((-1, -2))
        collision = masked_mean(F.relu(cfg.width+.2-distance).square().mean(-1), pair_mask)
        side = torch.stack([-direction[..., 1], direction[..., 0]], -1)
        if polygon is not None:
            footprint = torch.stack([positions+a*cfg.length/2*direction+b*cfg.width/2*side
                                     for a in (-1, 0, 1) for b in (-1, 0, 1)], dim=-2)
            grid = 2*(footprint-lo)/(hi-lo)-1
            clearance = F.grid_sample(sdf, grid.reshape(1, -1, 1, 2), align_corners=True,
                                      padding_mode='border').reshape(footprint.shape[:-1])
            road = masked_mean(F.relu(.1-clearance).square().amax(-1).mean(-1), active)
        else:
            nearest = (positions[..., None, :]-center).square().sum(-1).argmin(-1)
            mid = (lower[nearest]+upper[nearest])/2
            chord = upper[nearest]-lower[nearest]
            widths = chord.norm(dim=-1).clamp_min(1e-6)
            normal = chord/widths[..., None]
            radius = (direction*normal).sum(-1).abs()*cfg.length/2+(side*normal).sum(-1).abs()*cfg.width/2
            lateral = ((positions-mid)*normal).sum(-1).abs()
            road = masked_mean(F.relu(lateral+radius+.1-widths/2).square().mean(-1), active)
        goal = masked_mean((positions[..., -1, :]-goals[..., :2]).norm(dim=-1)/cfg.pos_scale, active)
        return 10*collision+10*road+goal
    return cost


def make_model(name, cfg):
    if name == 'ctrl_sim':
        return CtRLSim(cfg)
    if name == 'ctg_plus_plus':
        return CTGPlusPlus(cfg)
    raise ValueError('Unknown new baseline: '+name)
