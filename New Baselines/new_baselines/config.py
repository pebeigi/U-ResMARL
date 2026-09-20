from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace as NS


@dataclass
class Config:
    decision_protocol_version: int = 1
    perception_radius: float = 60.
    max_neighbors: int = 6
    dt: float = .5
    max_agents: int = 24
    context: int = 12
    history: int = 3
    horizon: int = 10
    hidden: int = 256
    layers: int = 2
    decoder_layers: int = 4
    diffusion_steps: int = 100
    accel_bins: int = 20
    steer_bins: int = 50
    return_bins: int = 350
    max_accel: float = 4.
    max_steer: float = .45
    max_speed: float = 16.
    wheelbase: float = 2.8
    steer_from_rest: bool = False
    min_steer_speed: float = .5
    length: float = 4.5
    width: float = 1.8
    map_segments: int = 12
    map_points: int = 20
    pos_scale: float = 100.
    vel_scale: float = 40.
    # Validation-tuned inference parameters belong in the saved checkpoint.
    goal_tilt: float = 0.
    vehicle_tilt: float = 0.
    road_tilt: float = 0.
    guidance_scale: float = 0.

    def validate(self):
        if self.dt <= 0 or self.max_agents < 1 or self.hidden % 8 or self.hidden < 8:
            raise ValueError('Positive dt/agents and hidden dimension divisible by 8 required')
        if min(self.context, self.history, self.horizon, self.diffusion_steps) < 2:
            raise ValueError('Context, history, horizon and diffusion steps must be >= 2')
        if self.history + self.horizon > 100:
            raise ValueError('Upstream positional encoding supports at most 100 steps')
        if self.context > 512 or self.map_segments < 3 or self.map_segments % 3 or self.map_points < 2:
            raise ValueError('Context <=512, map segments divisible by 3 and >=2 map points required')
        if min(self.accel_bins, self.steer_bins, self.return_bins) < 2:
            raise ValueError('Each discretization needs at least two bins')
        if min(self.max_accel, self.max_steer, self.max_speed, self.wheelbase,
               self.length, self.width, self.pos_scale, self.vel_scale) <= 0:
            raise ValueError('Dynamics dimensions, action limits and normalization scales must be positive')
        return self

    def upstream(self, diffusion=False):
        self.validate()
        waymo = NS(max_num_agents=self.max_agents, train_context_length=(self.history+self.horizon if diffusion else self.context),
                   input_horizon=self.history, max_timestep=512, accel_discretization=self.accel_bins,
                   steer_discretization=self.steer_bins, rtg_discretization=self.return_bins,
                   max_num_road_polylines=self.map_segments, max_num_road_pts_per_polyline=self.map_points,
                   goal_dim=5, k_attr=7, num_agent_types=5, num_road_types=8, map_attr=2, action_dim=2)
        model = NS(hidden_dim=self.hidden, num_heads=8, dim_feedforward=4*self.hidden,
                   num_transformer_encoder_layers=self.layers, num_decoder_layers=self.decoder_layers,
                   dropout=.1, state_dim=12, map_attr=3, num_road_types=8, use_map=True,
                   encode_initial_state=True, decision_transformer=False, trajeglish=False, il=False,
                   no_actions=False, num_reward_components=3, goal_dropout=.1, predict_rtg=True,
                   predict_future_states=True, attend_own_return_action=False,
                   diffusion_type='states_actions', use_rtg=False)
        return NS(model=model, dataset=NS(waymo=waymo))

    def to_dict(self):
        return asdict(self)
