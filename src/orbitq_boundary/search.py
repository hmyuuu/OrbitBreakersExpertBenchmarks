"""A simple, auditable coordinate-wise boundary-search policy."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .comparison import assert_controlled_mutation
from .models import (
    BoundarySearchState,
    CandidateTaskSpec,
    CandidateVerdict,
    FormalScaleObservation,
    NonMonotonicObservation,
    TaskFamilySpec,
    canonical_json_bytes,
)
from .specs import generate_candidate


@dataclass(frozen=True)
class SearchDecision:
    candidate: CandidateTaskSpec | None
    state: BoundarySearchState
    reason: str


def _parameter_vector_key(parameters) -> str:
    return canonical_json_bytes(parameters).decode("utf-8")


def _validate_search_inputs(
    family: TaskFamilySpec,
    current: CandidateTaskSpec,
    state: BoundarySearchState,
    verdict: CandidateVerdict,
) -> None:
    if state.family_id != family.family_id or current.family_id != family.family_id:
        raise ValueError("search family does not match current candidate")
    if state.current_candidate_id != current.candidate_id:
        raise ValueError("search state does not reference current candidate")
    if verdict.candidate_id != current.candidate_id:
        raise ValueError("verdict does not belong to current candidate")
    if (
        verdict.agent_config_fingerprint
        != state.agent_config_fingerprint
    ):
        raise ValueError("verdict belongs to a different AgentConfig")
    declared = {parameter.name for parameter in family.parameters}
    if not state.axis_order or len(set(state.axis_order)) != len(state.axis_order):
        raise ValueError("axis_order must contain unique parameter names")
    unknown = set(state.axis_order) - declared
    if unknown:
        raise ValueError(f"axis_order contains unknown parameters: {sorted(unknown)}")
    if not 0 <= state.axis_index < len(state.axis_order):
        raise ValueError("axis_index is outside axis_order")


def select_next_candidate(
    family: TaskFamilySpec,
    current: CandidateTaskSpec,
    state: BoundarySearchState,
    verdict: CandidateVerdict,
    generation_seed: int,
) -> SearchDecision:
    """Select one controlled local mutation or return a terminal state."""

    _validate_search_inputs(family, current, state, verdict)
    if verdict.checkpoint_kind == "screening":
        return SearchDecision(
            candidate=None,
            state=replace(state, status="awaiting-formal-checkpoint"),
            reason="screening labels cannot advance the scale search",
        )
    axis_name = state.axis_order[state.axis_index]
    by_name = {parameter.name: parameter for parameter in family.parameters}
    parameter = by_name[axis_name]
    current_value = current.parameters[axis_name]
    try:
        current_index = parameter.values.index(current_value)
    except ValueError as exc:
        raise ValueError(f"current value is outside domain for {axis_name}") from exc

    observation = FormalScaleObservation(
        candidate_id=current.candidate_id,
        parameter_name=axis_name,
        parameter_value=current_value,
        endpoint_classification=verdict.endpoint_classification,
        pass_rate=verdict.pass_rate,
    )
    observations = (
        *state.formal_scale_observations,
        observation,
    )
    non_monotonic = list(state.non_monotonic_observations)
    for prior in state.formal_scale_observations:
        if prior.parameter_name != axis_name:
            continue
        try:
            prior_index = parameter.values.index(prior.parameter_value)
        except ValueError:
            continue
        if parameter.harder_direction == "ascending":
            prior_is_easier = prior_index < current_index
        else:
            prior_is_easier = prior_index > current_index
        if prior_is_easier and verdict.pass_rate > prior.pass_rate:
            non_monotonic.append(
                NonMonotonicObservation(
                    easier_candidate_id=prior.candidate_id,
                    harder_candidate_id=current.candidate_id,
                    parameter_name=axis_name,
                    easier_pass_rate=prior.pass_rate,
                    harder_pass_rate=verdict.pass_rate,
                    notes="harder formal observation has a higher pass rate",
                )
            )
    observed_state = replace(
        state,
        formal_scale_observations=observations,
        non_monotonic_observations=tuple(non_monotonic),
    )
    if verdict.endpoint_classification == "hard":
        lower_easy_exists = False
        for prior in state.formal_scale_observations:
            if (
                prior.parameter_name != axis_name
                or prior.endpoint_classification != "easy"
            ):
                continue
            try:
                prior_index = parameter.values.index(prior.parameter_value)
            except ValueError:
                continue
            if parameter.harder_direction == "ascending":
                lower_easy_exists = prior_index < current_index
            else:
                lower_easy_exists = prior_index > current_index
            if lower_easy_exists:
                break
        if lower_easy_exists:
            return SearchDecision(
                candidate=None,
                state=replace(observed_state, status="bracketed"),
                reason=(
                    "formal easy and higher formal hard bracket the same "
                    f"ordered axis {axis_name}"
                ),
            )
        if parameter.harder_direction == "ascending":
            easier_indices = range(current_index - 1, -1, -1)
        else:
            easier_indices = range(current_index + 1, len(parameter.values))
        visited = set(state.visited_parameter_vectors)
        for candidate_index in easier_indices:
            new_parameters = dict(current.parameters)
            new_parameters[axis_name] = parameter.values[candidate_index]
            vector_key = _parameter_vector_key(new_parameters)
            if vector_key in visited:
                continue
            candidate = generate_candidate(
                family,
                new_parameters,
                generation_seed,
                parent=current,
                changed_parameter=axis_name,
            )
            assert_controlled_mutation(family, current, candidate)
            return SearchDecision(
                candidate=candidate,
                state=replace(
                    observed_state,
                    current_candidate_id=candidate.candidate_id,
                    visited_parameter_vectors=(
                        *state.visited_parameter_vectors,
                        vector_key,
                    ),
                    status="need-easier-anchor",
                ),
                reason=(
                    "formal hard has no lower formal easy; selected an "
                    f"easier anchor on axis {axis_name}"
                ),
            )
        return SearchDecision(
            candidate=None,
            state=replace(observed_state, status="easier-anchor-unavailable"),
            reason="formal hard has no lower formal easy and no easier candidate",
        )
    if verdict.endpoint_classification == "unresolved":
        return SearchDecision(
            candidate=None,
            state=replace(
                observed_state,
                status="awaiting-approved-sampling-policy",
            ),
            reason=(
                "formal endpoint is unresolved; additional sampling requires "
                "an approved budget and valid sequential error control"
            ),
        )
    if verdict.endpoint_classification != "easy":
        raise ValueError(
            "unsupported endpoint classification: "
            f"{verdict.endpoint_classification}"
        )

    visited = set(state.visited_parameter_vectors)
    for axis_index in range(state.axis_index, len(state.axis_order)):
        axis_name = state.axis_order[axis_index]
        parameter = by_name[axis_name]
        current_value = current.parameters[axis_name]
        try:
            current_index = parameter.values.index(current_value)
        except ValueError as exc:
            raise ValueError(
                f"current value is outside domain for {axis_name}"
            ) from exc
        if parameter.harder_direction == "ascending":
            candidate_indices = range(current_index + 1, len(parameter.values))
        else:
            candidate_indices = range(current_index - 1, -1, -1)
        for candidate_index in candidate_indices:
            new_parameters = dict(current.parameters)
            new_parameters[axis_name] = parameter.values[candidate_index]
            vector_key = _parameter_vector_key(new_parameters)
            if vector_key in visited:
                continue
            candidate = generate_candidate(
                family,
                new_parameters,
                generation_seed,
                parent=current,
                changed_parameter=axis_name,
            )
            assert_controlled_mutation(family, current, candidate)
            tested = state.tested_candidate_ids
            if current.candidate_id not in tested:
                tested = (*tested, current.candidate_id)
            next_state = replace(
                observed_state,
                axis_index=axis_index,
                current_candidate_id=candidate.candidate_id,
                visited_parameter_vectors=(
                    *state.visited_parameter_vectors,
                    vector_key,
                ),
                tested_candidate_ids=tested,
                status="candidate-selected",
            )
            return SearchDecision(
                candidate=candidate,
                state=next_state,
                reason=f"selected harder neighbor on axis {axis_name}",
            )
    return SearchDecision(
        candidate=None,
        state=replace(state, status="exhausted"),
        reason="no unvisited harder neighbor remains",
    )


def record_non_monotonic_observation(
    state: BoundarySearchState,
    easier_candidate_id: str,
    harder_candidate_id: str,
    parameter_name: str,
    easier_pass_rate: float,
    harder_pass_rate: float,
    notes: str,
) -> BoundarySearchState:
    """Append a non-monotonic result without altering prior evidence."""

    if not easier_candidate_id or not harder_candidate_id:
        raise ValueError("candidate ids must be non-empty")
    if not parameter_name:
        raise ValueError("parameter_name must be non-empty")
    if not 0.0 <= easier_pass_rate <= 1.0:
        raise ValueError("easier_pass_rate must be between zero and one")
    if not 0.0 <= harder_pass_rate <= 1.0:
        raise ValueError("harder_pass_rate must be between zero and one")
    observation = NonMonotonicObservation(
        easier_candidate_id=easier_candidate_id,
        harder_candidate_id=harder_candidate_id,
        parameter_name=parameter_name,
        easier_pass_rate=easier_pass_rate,
        harder_pass_rate=harder_pass_rate,
        notes=notes,
    )
    return replace(
        state,
        non_monotonic_observations=(
            *state.non_monotonic_observations,
            observation,
        ),
    )
