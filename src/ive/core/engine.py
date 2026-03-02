"""
IVE Engine — Main Orchestrator.

The IVEEngine is the top-level coordinator that drives the four-phase
pipeline: Understand → Model → Detect → Construct.

Each phase is run sequentially so that outputs from earlier phases can
be consumed by later ones. The engine publishes progress events to Redis
after each phase completes so that the WebSocket handler can relay them
to connected clients.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog

from ive.core.phase_construct import PhaseConstruct
from ive.core.phase_detect import PhaseDetect
from ive.core.phase_model import PhaseModel
from ive.core.phase_understand import PhaseUnderstand
from ive.core.pipeline import EngineResult, PipelineContext

if TYPE_CHECKING:
    from ive.api.v1.schemas.experiment_schemas import ExperimentConfig

log = structlog.get_logger(__name__)


class IVEEngine:
    """
    Orchestrates the four-phase Invisible Variables Engine pipeline.

    Usage:
        engine = IVEEngine()
        result = await engine.run(experiment_id, config)

    The engine is stateless — a new instance can be created per Celery task.
    All state is passed through the PipelineContext dataclass.
    """

    def __init__(self) -> None:
        """Initialise the engine and register pipeline phases."""
        self._phases = [
            PhaseUnderstand(),
            PhaseModel(),
            PhaseDetect(),
            PhaseConstruct(),
        ]

    async def run(
        self,
        experiment_id: uuid.UUID,
        config: ExperimentConfig,
        data_path: str,
    ) -> EngineResult:
        """
        Execute the full IVE pipeline for an experiment.

        Args:
            experiment_id: UUID of the experiment being run.
            config: ExperimentConfig controlling model and pipeline parameters.
            data_path: Path to the dataset artifact on disk.

        Returns:
            EngineResult containing all discovered latent variables.

        TODO:
            - Load dataset from ArtifactStore using data_path
            - Create PipelineContext with loaded data
            - Iterate phases and call _run_phase()
            - Publish Redis progress events after each phase
            - Return EngineResult
        """
        log.info("ive.engine.start", experiment_id=str(experiment_id))
        start_time = time.monotonic()

        ctx = PipelineContext(
            experiment_id=experiment_id,
            config=config,
            data_path=data_path,
        )

        for phase in self._phases:
            phase_name = phase.get_phase_name()
            log.info("ive.engine.phase_start", phase=phase_name)

            try:
                phase_result = await self._run_phase(phase, ctx)
                ctx.phase_results[phase_name] = phase_result
                log.info("ive.engine.phase_complete", phase=phase_name)

                # TODO: Publish progress event to Redis pub/sub
                # await self._publish_progress(experiment_id, phase_name, progress_pct)

            except Exception as exc:
                log.error(
                    "ive.engine.phase_failed",
                    phase=phase_name,
                    error=str(exc),
                    experiment_id=str(experiment_id),
                )
                raise

        elapsed = time.monotonic() - start_time
        log.info("ive.engine.complete", elapsed_s=round(elapsed, 2))

        return EngineResult(
            experiment_id=experiment_id,
            latent_variables=ctx.latent_variables,
            elapsed_seconds=elapsed,
        )

    async def _run_phase(self, phase: object, ctx: PipelineContext) -> object:
        """
        Execute a single phase and return its result.

        Wraps phase execution with timing, logging, and error normalisation.

        TODO:
            - Add phase-level retry logic with tenacity
            - Record phase timing in ctx
        """
        # TODO: Import and use the PhaseBase type annotation properly
        return await phase.execute(ctx)  # type: ignore[union-attr]

    async def _publish_progress(
        self,
        experiment_id: uuid.UUID,
        phase: str,
        progress_pct: int,
        message: str = "",
    ) -> None:
        """
        Publish a progress event to Redis for WebSocket relay.

        TODO:
            - Import Redis client
            - Publish JSON payload to channel f"ive:progress:{experiment_id}"
        """
        # TODO: Implement Redis pub
        pass
