"""Synthetic full-agent loop proving Stage 0 interfaces and control ordering."""

from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn

from bus.message import BusMessage, MessageHeader
from bus.schemas import MessageType, validate_message
from control.mpc import choose_candidates, score_action_candidates
from control.safety import SafetyLimits
from models.body_dynamics import TinyBodyDynamics
from models.executive import TinyExecutive
from models.jepa_encoder import TinyJEPAEncoder
from models.jepa_world import TinyJEPAWorldPredictor
from models.language_speech import ByteTokenizer, TinyLanguageSpeech
from models.memory import InMemoryStore, TinyMemoryController
from models.motor_cortex import TinyMotorCortex
from models.orchestrator import MessageRouter, TinyOrchestrator
from sim.mock_robot import MockSO101
from training.seed import seed_everything


@dataclass(frozen=True)
class Stage0SystemConfig:
    bus_dim: int = 32
    d_model: int = 32
    num_heads: int = 4
    image_size: int = 32
    world_tokens: int = 4
    horizon: int = 4
    candidates: int = 2
    flow_steps: int = 2
    execute_prefix: int = 2
    seed: int = 7


@dataclass
class Stage0StepResult:
    initial_state: Tensor
    final_state: Tensor
    q_goal: Tensor
    candidate_costs: Tensor
    selected_candidate: Tensor
    executed_actions: Tensor
    prediction_residual: Tensor
    messages: dict[str, str]
    memory_entries: int
    route_logits: Tensor
    revised_q_goal: Tensor


class Stage0RobotBrain(nn.Module):
    """CPU-sized composition of all eight modules and deterministic control."""

    def __init__(self, config: Stage0SystemConfig = Stage0SystemConfig()) -> None:
        super().__init__()
        seed_everything(config.seed)
        self.config = config
        common = {
            "d_model": config.d_model,
            "num_heads": config.num_heads,
        }
        self.language = TinyLanguageSpeech(
            bus_dim=config.bus_dim,
            semantic_depth=1,
            conformer_depth=1,
            **common,
        )
        self.jepa_encoder = TinyJEPAEncoder(
            image_size=config.image_size,
            patch_size=8,
            max_views=2,
            world_tokens=config.world_tokens,
            bus_dim=config.bus_dim,
            depth=1,
            **common,
        )
        self.executive = TinyExecutive(
            bus_dim=config.bus_dim,
            depth=1,
            **common,
        )
        self.motor = TinyMotorCortex(
            horizon=config.horizon,
            depth=1,
            **common,
        )
        self.body = TinyBodyDynamics(depth=1, **common)
        self.world = TinyJEPAWorldPredictor(
            bus_dim=config.bus_dim,
            depth=1,
            **common,
        )
        self.memory = TinyMemoryController(
            bus_dim=config.bus_dim,
            d_model=config.d_model,
            top_k=2,
        )
        self.orchestrator = TinyOrchestrator(
            bus_dim=config.bus_dim,
            hidden_dim=config.d_model,
        )
        self.tokenizer = ByteTokenizer()
        self.memory_store = InMemoryStore(config.bus_dim)
        self.robot = MockSO101()
        self.message_router = MessageRouter(bus_dim=config.bus_dim, max_age_seconds=5.0)
        self.message_router.register("executive", lambda _: None)
        self._sequence = 0
        self._control_cycle = 0
        self.eval()

    @torch.no_grad()
    def step(self, command: str, images: Tensor | None = None) -> Stage0StepResult:
        config = self.config
        if images is None:
            images = torch.zeros(1, 1, 3, config.image_size, config.image_size)
        if images.shape[0] != 1:
            raise ValueError("the Stage 0 demo currently runs one mock robot")
        initial_state = self.robot.state().clone()
        motor_state = self.robot.motor_state()

        text_ids, text_valid = self.tokenizer.encode([command])
        language = self.language(text_ids=text_ids, text_valid=text_valid)
        task_message = self._message(
            MessageType.TASK_GOAL,
            "language",
            "executive",
            {
                "semantic_token": language.semantic_token,
                "intent_id": language.intent_logits.argmax(dim=-1),
            },
            metadata={"text": command},
        )
        self.message_router.route(task_message, now=task_message.header.timestamp)

        perception = self.jepa_encoder(images, motor_state)
        world_state_message = self._message(
            MessageType.WORLD_STATE,
            "jepa_encoder",
            "executive",
            {
                "world_tokens": perception.world_tokens,
                "robot_state": motor_state,
            },
        )
        validate_message(world_state_message, bus_dim=config.bus_dim)

        memory_tokens = None
        if len(self.memory_store):
            entries, recency, confidence = self.memory_store.tensors()
            recalled = self.memory(
                language.semantic_token,
                entries.unsqueeze(0),
                recency.unsqueeze(0),
                confidence.unsqueeze(0),
            )
            memory_tokens = recalled.retrieved_entries

        executive = self.executive(
            language.semantic_token,
            perception.world_tokens,
            initial_state,
            memory_tokens=memory_tokens,
        )
        motor_goal = executive.motor_goal_tensor()
        motor_goal_message = self._message(
            MessageType.MOTOR_GOAL,
            "executive",
            "motor_cortex",
            {
                "q_goal": executive.q_goal,
                "qdot_goal": executive.qdot_goal,
                "duration": executive.duration,
                "constraints": executive.constraints,
            },
        )
        validate_message(motor_goal_message, bus_dim=config.bus_dim)

        generator = torch.Generator(device=initial_state.device)
        generator.manual_seed(config.seed + self._control_cycle)
        motor = self.motor.sample(
            motor_state,
            motor_goal,
            candidates=config.candidates,
            steps=config.flow_steps,
            goal_guidance=1.0,
            generator=generator,
        )
        action_message = self._message(
            MessageType.ACTION_CANDIDATES,
            "motor_cortex",
            "body_world",
            {"actions": motor.actions},
        )
        validate_message(action_message, bus_dim=config.bus_dim)

        body = self.body(initial_state, motor.actions)
        world = self.world(perception.world_tokens, motor.actions)
        body_message = self._message(
            MessageType.BODY_PREDICTION,
            "body_dynamics",
            "scorer",
            {"mean": body.mean, "log_variance": body.log_variance},
        )
        world_prediction_message = self._message(
            MessageType.WORLD_PREDICTION,
            "jepa_world",
            "scorer",
            {
                "future_tokens": world.future_tokens,
                "event_logits": world.event_logits,
            },
        )
        validate_message(body_message, bus_dim=config.bus_dim)
        validate_message(world_prediction_message, bus_dim=config.bus_dim)

        limits = SafetyLimits.conservative_stage0()
        event_probability = world.event_logits.sigmoid()
        costs = score_action_candidates(
            actions=motor.actions,
            body_mean=body.mean,
            body_log_variance=body.log_variance,
            q_goal=executive.q_goal,
            qdot_goal=executive.qdot_goal,
            joint_min=limits.joint_min,
            joint_max=limits.joint_max,
            collision_probability=event_probability[..., 2],
            task_cost=1.0 - event_probability[..., 1],
        )
        selected_actions, selected_indices = choose_candidates(motor.actions, costs)
        safe_actions = self.robot.safety.filter_chunk(self.robot.q, selected_actions)
        prefix = safe_actions[:, : config.execute_prefix]
        final_state = self.robot.execute_relative(prefix).clone()

        batch_indices = torch.arange(body.mean.shape[0])
        predicted_at_prefix = body.mean[
            batch_indices, selected_indices, config.execute_prefix - 1
        ]
        residual = final_state - predicted_at_prefix
        execution_message = self._message(
            MessageType.EXECUTION_RESULT,
            "robot_stack",
            "executive_memory",
            {
                "actual_state": final_state,
                "prediction_residual": residual,
            },
            metadata={"hard_limit_event": "false"},
        )
        validate_message(execution_message, bus_dim=config.bus_dim)

        self.memory_store.write(
            executive.next_stage_latent[0],
            confidence=float(motor.confidence[0, selected_indices[0]]),
            kind="execution",
            metadata={"command": command},
            timestamp=time.time(),
        )
        entries, recency, confidence = self.memory_store.tensors()
        self.memory(
            language.semantic_token,
            entries.unsqueeze(0),
            recency.unsqueeze(0),
            confidence.unsqueeze(0),
        )
        routing_features = torch.tensor([[0.5, 0.0, task_message.header.confidence]])
        route_logits = self.orchestrator(language.semantic_token, routing_features)

        revised = self.executive(
            language.semantic_token,
            perception.world_tokens,
            final_state,
            memory_tokens=entries.unsqueeze(0),
            body_predictions=body.mean,
            world_event_logits=world.event_logits,
        )
        self._control_cycle += 1
        messages = {
            "task_goal": task_message.to_json(),
            "world_state": world_state_message.to_json(),
            "motor_goal": motor_goal_message.to_json(),
            "action_candidates": action_message.to_json(),
            "body_prediction": body_message.to_json(),
            "world_prediction": world_prediction_message.to_json(),
            "execution_result": execution_message.to_json(),
        }
        return Stage0StepResult(
            initial_state=initial_state,
            final_state=final_state,
            q_goal=executive.q_goal,
            candidate_costs=costs,
            selected_candidate=selected_indices,
            executed_actions=prefix,
            prediction_residual=residual,
            messages=messages,
            memory_entries=len(self.memory_store),
            route_logits=route_logits,
            revised_q_goal=revised.q_goal,
        )

    def _message(
        self,
        message_type: MessageType,
        source: str,
        destination: str,
        tensors: dict[str, Tensor],
        *,
        metadata: dict[str, str] | None = None,
    ) -> BusMessage:
        self._sequence += 1
        return BusMessage(
            header=MessageHeader.create(
                message_type,
                source,
                destination,
                self._sequence,
                timestamp=time.time(),
            ),
            tensors=tensors,
            metadata=metadata or {},
        )
