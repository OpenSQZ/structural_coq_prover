# signs for coq
signs = [ "forall", "fun", "match", "in", "for",
    "end", "as", "let", "if", "then", "else", "return",
    "SProp", "Prop", "Set", "Type", "all:", "with", "id",
    "(", ")", "<-", "[>", "|-", "]:", "[", "]", "{", "}", "*",
    ":=", "=>", "->", "..", "<:", "<<:", ":>", "|",
    ".(", "()", "`{", "`(", "@{", "{|",
    "_", "@", "+", "!", "?", ";", ",", ":" ]

# TODO: need process in coqc
HintDb = ['core', 'program', 'discriminated', 'typeclass_instances']

goal_completed = "goalcompleted"

ANONYMOUS_IDENTIFIER = "_Anonymous"

Unnamed_thm = "Unnamed_thm"

Rels = [f"Rel({i+1})" for i in range(16)]

num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

internal_tactic = [
      "intro", "intros", "apply", "rewrite", "refine", "case", "clear", "injection", "eintros",
      "progress", "setoid_rewrite", "left", "right", "constructor",
      "econstructor", "decide equality", "abstract", "exists", "cbv", "simple destruct",
      "info", "field", "specialize", "evar", "solve", "instantiate", "info_auto", "info_eauto",
      "quote", "eexact", "autorewrite",
      "destruct", "destruction", "destruct_call", "dependent", "elim", "extensionality",
      "f_equal", "generalize", "generalize_eqs", "generalize_eqs_vars", "induction", "rename", "move",
      "set", "assert", "do", "repeat",
      "cut", "assumption", "exact", "split", "subst", "try", "discriminate",
      "simpl", "unfold", "red", "compute", "at", "by",
      "reflexivity", "symmetry", "transitivity",
      "replace", "setoid_replace", "inversion", "inversion_clear",
      "pattern", "intuition", "congruence", "fail", "fresh",
      "trivial", "tauto", "firstorder", "ring",
      "clapply", "program_simpl", "program_simplify", "eapply", "auto", "eauto",
      "change", "fold", "hnf", "lazy", "simple", "eexists", "debug", "idtac", "first", "type of", "pose",
      "eval", "instantiate", "until", "now",
      "cbn", "dauto", "Numeral", "iExists", "asserts", "revert", "astepr", "etransitivity", "Presentation",
      "clearbody", "if_tac", "Step_final", "typeclasses", "algebra", "erewrite", "wp_apply", "Identity", "typeclass",
      "transparent", "set_solver", "Create", "wp_lam", "unshelve", "iMod", "forget", "Canonical", "astepl", "snrapply",
      "Register", "iIntros", "rtype_equalizer", "iModIntro", "have", "Adjoint", "contradiction", "field_simplify_eq",
      "intros_all", "applys", "suff", "pols", "funext", "iSplitL", "polr", "nrefine", "decompose", "etrans", "polf",
      "extens", "wp_let", "remember", "iApply", "Polymorphic", "fequals", "lapply", "Existing", "hyp_polf", "qeauto",
      "introv", "field_simplify", "icase", "nrapply", "iFrame", "Universe", "srapply", "eassumption", "iDestruct",
      "suffices", "srefine", "iSplit", "Admitted", "autoapply", "head_of_constr", "flatten_contravariant_conj", 
      "with_uniform_flags", "elimtype", "nzinduct", "absurd", "flatten_contravariant_disj", "change_no_check", "with_power_flags"
]

internal_glob = []
for sign in signs:
    internal_glob.append(sign)
internal_glob.append(ANONYMOUS_IDENTIFIER)
internal_glob.append(Unnamed_thm)
internal_glob.append(goal_completed)
internal_glob.extend(Rels)
internal_glob.extend(internal_tactic)
internal_glob.extend(num)
internal_glob.extend(HintDb)
