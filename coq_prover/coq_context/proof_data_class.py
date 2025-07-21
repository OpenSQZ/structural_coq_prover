from data_extraction.coq_data.Ps_class import PSItem, ps_object_single, State
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
from enum import Enum


class TacticStatus(Enum):
    NORMAL = "normal"
    SUCCESS = "success"
    FAIL = "fail"
    GIVEUP = "giveup"
    SUBCOM = "subcompleted"
    COMPLETED = "completed"

    def to_value(self):
        return self.value

@dataclass
class TacticTrace_Single:
    tactic: str
    error_message: str = None
    reason: str = None

    def to_dict(self):
        return {
            "tactic": self.tactic,
            "error_message": self.error_message,
            "reason": self.reason
        }

@dataclass
class TacticResult:
    status: TacticStatus
    ps_item: PSItem
    tactic_sequence: List[str]
    tactic_trace: List[TacticTrace_Single] = field(default_factory=list)
    error_message: str = None

@dataclass
class TacticResultGroup:
    tactic_results: List[TacticResult]
    status: TacticStatus = TacticStatus.NORMAL
    all_success: bool = False
    completed_results: TacticResult = None
    subcompleted_results: TacticResult = None

    ## for resonsider, so all success will false default
    def merge(self, other: 'TacticResultGroup'):
        self.tactic_results.extend(other.tactic_results)
        self.all_success = False
        if other.status == TacticStatus.COMPLETED:
            self.status = TacticStatus.COMPLETED
            self.completed_results = other.completed_results
        elif other.status == TacticStatus.SUBCOM:
            self.status = TacticStatus.SUBCOM
            self.subcompleted_results = other.subcompleted_results

@dataclass
class PreviousResult:
    ps_item: PSItem
    tactic_sequence: List[str]

@dataclass
class RemainingState:
    states: List[State]
    tactic_result: TacticResult
    path: str

@dataclass
class ProofTrace:
    brief_strategy: str
    state_explanation: Dict
    tactic: str

@dataclass
class ProofSummaryWithTactic:
    proof_summary: Dict
    tactic: List[str]

@dataclass
class ProofInfo:
    ## when init is PSItem, else is ps_object_single
    curr_ps: Union[PSItem, ps_object_single] = None
    # previous result, including ps_item and tactic_sequence
    prev_result: PreviousResult = None
    curr_path: str = '1/1'
    # remaining list, including remaining states, tactic result and path
    remaining_list: List[RemainingState] = field(default_factory=list)
    # proof traces, including brief strategy, state intuition and tactic
    proof_traces: List[ProofTrace] = field(default_factory=list)
    # proof summary with tactic, including proof summary and tactic
    proof_summary_with_tactic: ProofSummaryWithTactic = None
    is_finished: bool = False

@dataclass
class ProofContext:
    theorem_name: str
    theorem_path: str
    ps_init: PSItem
    public_notes: List[Tuple[str,str]] = field(default_factory=list)
    depth: int = 0
    log_file: str = None

@dataclass
class ProofInfoLayer:
    proof_context: ProofContext
    proof_infos: List[ProofInfo]

@dataclass
class GenerateInfo:
    prompt_tactic: str
    response_tactic: List[Union[str, Dict]]
    prompt_tactic_reorganize: str = None
    prompt_method: str = None
    response_method: str = None
    brief_strategy: str = None
    prompt_reconsider_tactic: str = None
    response_reconsider_tactic: str = None
    prompt_reconsider_method: str = None
    response_reconsider_method: str = None
    # method can be reconsider as well, now do not support
    retrieval_info: Dict = None
    
@dataclass
class SingleItemInfo:
    path: str
    ps_item: PSItem
    tactics: List[str]
    depth: int
    status: TacticStatus
    tactic_traces: List[TacticTrace_Single] = None
    proof_summary: Dict = None
    explanation: Dict = None
    extra_info: Dict = None

    def to_dict(self):
        return  {
            self.path:
            {
            "ps_item": self.ps_item.to_dict() if self.ps_item.Content != None else {},
            "tactics": self.tactics,
            "depth": self.depth,
            "status": self.status.to_value(),
            "tactic_traces": [tactic_trace.to_dict() for tactic_trace in self.tactic_traces],
            "proof_summary": self.proof_summary,
            "explanation": self.explanation,
            "extra_info": self.extra_info
            }
        }
    
@dataclass
class LogInfo:
    prompt_response_info: Dict
    items_info: List[SingleItemInfo] = field(default_factory=list)

    def to_dict(self):
        return{
            "prompt_response_info": self.prompt_response_info,
            # "items_info": {**{item.to_dict() for item in self.items_info}}
            "items_info": {k: v for d in self.items_info for k, v in d.to_dict().items()}
        }