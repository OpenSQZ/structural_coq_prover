(************************************************************************)
(*         *   The Coq Proof Assistant / The Coq Development Team       *)
(*  v      *   INRIA, CNRS and contributors - Copyright 1999-2019       *)
(* <O___,, *       (see CREDITS file for the list of authors)           *)
(*   \VV/  **************************************************************)
(*    //   *    This file is distributed under the terms of the         *)
(*         *     GNU Lesser General Public License Version 2.1          *)
(*         *     (see LICENSE file for the text of the license)         *)
(************************************************************************)

(* This file contains the syntax-directed part of the type inference
   algorithm introduced by Murthy in Coq V5.10, 1995; the type
   inference algorithm was initially developed in a file named trad.ml
   which formerly contained a simple concrete-to-abstract syntax
   translation function introduced in CoC V4.10 for implementing the
   "exact" tactic, 1989 *)
(* Support for typing term in Ltac environment by David Delahaye, 2000 *)
(* Type inference algorithm made a functor of the coercion and
   pattern-matching compilation by Matthieu Sozeau, March 2006 *)
(* Fixpoint guard index computation by Pierre Letouzey, July 2007 *)

(* Structural maintainer: Hugo Herbelin *)
(* Secondary maintenance: collective *)


open Pp
open CErrors
open Util
open Names
open Evd
open Constr
open Context
open Termops
open Environ
open EConstr
open Vars
open Reductionops
open Type_errors
open Typing
open Evarutil
open Evardefine
open Pretype_errors
open Glob_term
open Glob_ops
open GlobEnv
open Evarconv
open Univ
(* open Prettyp *)

module NamedDecl = Context.Named.Declaration

type typing_constraint = OfType of types | IsType | WithoutTypeConstraint

let (!!) env = GlobEnv.env env

let bidi_hints =
  Summary.ref (GlobRef.Map.empty : int GlobRef.Map.t) ~name:"bidirectionalityhints"

let add_bidirectionality_hint gr n =
  bidi_hints := GlobRef.Map.add gr n !bidi_hints

let get_bidirectionality_hint gr =
  GlobRef.Map.find_opt gr !bidi_hints

let clear_bidirectionality_hint gr =
  bidi_hints := GlobRef.Map.remove gr !bidi_hints

(************************************************************************)
(* This concerns Cases *)
open Inductive
open Inductiveops

(************************************************************************)

(* An auxiliary function for searching for fixpoint guard indexes *)

exception Found of int array

let nf_fix sigma (nas, cs, ts) =
  let inj c = EConstr.to_constr ~abort_on_undefined_evars:false sigma c in
  (nas, Array.map inj cs, Array.map inj ts)

let search_guard ?loc env possible_indexes fixdefs =
  (* Standard situation with only one possibility for each fix. *)
  (* We treat it separately in order to get proper error msg. *)
  let is_singleton = function [_] -> true | _ -> false in
  if List.for_all is_singleton possible_indexes then
    let indexes = Array.of_list (List.map List.hd possible_indexes) in
    let fix = ((indexes, 0),fixdefs) in
    (try check_fix env fix
     with reraise ->
       let (e, info) = CErrors.push reraise in
       let info = Option.cata (fun loc -> Loc.add_loc info loc) info loc in
       iraise (e, info));
    indexes
  else
    (* we now search recursively among all combinations *)
    (try
       List.iter
         (fun l ->
            let indexes = Array.of_list l in
            let fix = ((indexes, 0),fixdefs) in
            (* spiwack: We search for a unspecified structural
               argument under the assumption that we need to check the
               guardedness condition (otherwise the first inductive argument
               will be chosen). A more robust solution may be to raise an
               error when totality is assumed but the strutural argument is
               not specified. *)
            try
              let flags = { (typing_flags env) with Declarations.check_guarded = true } in
              let env = Environ.set_typing_flags flags env in
              check_fix env fix; raise (Found indexes)
            with TypeError _ -> ())
         (List.combinations possible_indexes);
       let errmsg = "Cannot guess decreasing argument of fix." in
         user_err ?loc ~hdr:"search_guard" (Pp.str errmsg)
     with Found indexes -> indexes)

let esearch_guard ?loc env sigma indexes fix =
  let fix = nf_fix sigma fix in
  try search_guard ?loc env indexes fix
  with TypeError (env,err) ->
    raise (PretypeError (env,sigma,TypingError (map_ptype_error of_constr err)))

(* To force universe name declaration before use *)

let is_strict_universe_declarations =
  Goptions.declare_bool_option_and_ref
    ~depr:false
    ~name:"strict universe declaration"
    ~key:["Strict";"Universe";"Declaration"]
    ~value:true

(** Miscellaneous interpretation functions *)

let interp_known_universe_level_name evd qid =
  try
    let open Libnames in
    if qualid_is_ident qid then Evd.universe_of_name evd @@ qualid_basename qid
    else raise Not_found
  with Not_found ->
    let qid = Nametab.locate_universe qid in
    Univ.Level.make qid

let interp_universe_level_name ~anon_rigidity evd qid =
  try evd, interp_known_universe_level_name evd qid
  with Not_found ->
    if Libnames.qualid_is_ident qid then (* Undeclared *)
      let id = Libnames.qualid_basename qid in
      if not (is_strict_universe_declarations ()) then
        new_univ_level_variable ?loc:qid.CAst.loc ~name:id univ_rigid evd
      else user_err ?loc:qid.CAst.loc ~hdr:"interp_universe_level_name"
          (Pp.(str "Undeclared universe: " ++ Id.print id))
    else
      let dp, i = Libnames.repr_qualid qid in
      let num =
        try int_of_string (Id.to_string i)
        with Failure _ ->
          user_err ?loc:qid.CAst.loc ~hdr:"interp_universe_level_name"
            (Pp.(str "Undeclared global universe: " ++ Libnames.pr_qualid qid))
      in
      let level = Univ.Level.(make (UGlobal.make dp num)) in
      let evd =
        try Evd.add_global_univ evd level
        with UGraph.AlreadyDeclared -> evd
      in evd, level

let interp_universe_name ?loc evd l =
  (* [univ_flexible_alg] can produce algebraic universes in terms *)
  let anon_rigidity = univ_flexible in
  let evd', l = interp_universe_level_name ~anon_rigidity evd l in
  evd', l

let interp_sort_name ?loc sigma = function
  | GSProp -> sigma, Univ.Level.sprop
  | GProp -> sigma, Univ.Level.prop
  | GSet -> sigma, Univ.Level.set
  | GType l -> interp_universe_name ?loc sigma l

let interp_sort_info ?loc evd l =
    List.fold_left (fun (evd, u) (l,n) ->
      let evd', u' = interp_sort_name ?loc evd l in
      let u' = Univ.Universe.make u' in
      let u' = match n with
      | 0 -> u'
      | 1 -> Univ.Universe.super u'
      | n ->
        user_err ?loc ~hdr:"interp_universe"
          (Pp.(str "Cannot interpret universe increment +" ++ int n))
      in (evd', Univ.sup u u'))
    (evd, Univ.Universe.type0m) l

type inference_hook = env -> evar_map -> Evar.t -> evar_map * constr

type inference_flags = {
  use_typeclasses : bool;
  solve_unification_constraints : bool;
  fail_evar : bool;
  expand_evars : bool;
  program_mode : bool;
  polymorphic : bool;
}

(* Compute the set of still-undefined initial evars up to restriction
   (e.g. clearing) and the set of yet-unsolved evars freshly created
   in the extension [sigma'] of [sigma] (excluding the restrictions of
   the undefined evars of [sigma] to be freshly created evars of
   [sigma']). Otherwise said, we partition the undefined evars of
   [sigma'] into those already in [sigma] or deriving from an evar in
   [sigma] by restriction, and the evars properly created in [sigma'] *)

type frozen =
| FrozenId of evar_info Evar.Map.t
  (** No pending evars. We do not put a set here not to reallocate like crazy,
      but the actual data of the map is not used, only keys matter. All
      functions operating on this type must have the same behaviour on
      [FrozenId map] and [FrozenProgress (Evar.Map.domain map, Evar.Set.empty)] *)
| FrozenProgress of (Evar.Set.t * Evar.Set.t) Lazy.t
  (** Proper partition of the evar map as described above. *)

let frozen_and_pending_holes (sigma, sigma') =
  let undefined0 = Option.cata Evd.undefined_map Evar.Map.empty sigma in
  (* Fast path when the undefined evars where not modified *)
  if undefined0 == Evd.undefined_map sigma' then
    FrozenId undefined0
  else
    let data = lazy begin
    let add_derivative_of evk evi acc =
      match advance sigma' evk with None -> acc | Some evk' -> Evar.Set.add evk' acc in
    let frozen = Evar.Map.fold add_derivative_of undefined0 Evar.Set.empty in
    let fold evk _ accu = if not (Evar.Set.mem evk frozen) then Evar.Set.add evk accu else accu in
    let pending = Evd.fold_undefined fold sigma' Evar.Set.empty in
    (frozen, pending)
    end in
    FrozenProgress data

let apply_typeclasses ~program_mode env sigma frozen fail_evar =
  let filter_frozen = match frozen with
    | FrozenId map -> fun evk -> Evar.Map.mem evk map
    | FrozenProgress (lazy (frozen, _)) -> fun evk -> Evar.Set.mem evk frozen
  in
  let sigma = Typeclasses.resolve_typeclasses
      ~filter:(if program_mode
               then (fun evk evi -> Typeclasses.no_goals_or_obligations evk evi && not (filter_frozen evk))
               else (fun evk evi -> Typeclasses.no_goals evk evi && not (filter_frozen evk)))
      ~split:true ~fail:fail_evar env sigma in
  let sigma = if program_mode then (* Try optionally solving the obligations *)
      Typeclasses.resolve_typeclasses
        ~filter:(fun evk evi -> Typeclasses.all_evars evk evi && not (filter_frozen evk)) ~split:true ~fail:false env sigma
    else sigma in
  sigma

let apply_inference_hook hook env sigma frozen = match frozen with
| FrozenId _ -> sigma
| FrozenProgress (lazy (_, pending)) ->
  Evar.Set.fold (fun evk sigma ->
    if Evd.is_undefined sigma evk (* in particular not defined by side-effect *)
    then
      try
        let sigma, c = hook env sigma evk in
        Evd.define evk c sigma
      with Exit ->
        sigma
    else
      sigma) pending sigma

let apply_heuristics env sigma fail_evar =
  (* Resolve eagerly, potentially making wrong choices *)
  let flags = default_flags_of (Typeclasses.classes_transparent_state ()) in
  try solve_unif_constraints_with_heuristics ~flags env sigma
  with e when CErrors.noncritical e ->
    let e = CErrors.push e in
    if fail_evar then iraise e else sigma

let check_typeclasses_instances_are_solved ~program_mode env current_sigma frozen =
  (* Naive way, call resolution again with failure flag *)
  apply_typeclasses ~program_mode env current_sigma frozen true

let check_extra_evars_are_solved env current_sigma frozen = match frozen with
| FrozenId _ -> ()
| FrozenProgress (lazy (_, pending)) ->
  Evar.Set.iter
    (fun evk ->
      if not (Evd.is_defined current_sigma evk) then
        let (loc,k) = evar_source evk current_sigma in
        match k with
        | Evar_kinds.ImplicitArg (gr, (i, id), false) -> ()
        | _ ->
            error_unsolvable_implicit ?loc env current_sigma evk None) pending

(* [check_evars] fails if some unresolved evar remains *)

let check_evars env initial_sigma sigma c =
  let rec proc_rec c =
    match EConstr.kind sigma c with
    | Evar (evk, _) ->
      if not (Evd.mem initial_sigma evk) then
        let (loc,k) = evar_source evk sigma in
        begin match k with
          | Evar_kinds.ImplicitArg (gr, (i, id), false) -> ()
          | _ -> Pretype_errors.error_unsolvable_implicit ?loc env sigma evk None
        end
    | _ -> EConstr.iter sigma proc_rec c
  in proc_rec c

let check_evars_are_solved ~program_mode env sigma frozen =
  let sigma = check_typeclasses_instances_are_solved ~program_mode env sigma frozen in
  check_problems_are_solved env sigma;
  check_extra_evars_are_solved env sigma frozen

(* Try typeclasses, hooks, unification heuristics ... *)

let solve_remaining_evars ?hook flags env ?initial sigma =
  let program_mode = flags.program_mode in
  let frozen = frozen_and_pending_holes (initial, sigma) in
  let sigma =
    if flags.use_typeclasses
    then apply_typeclasses ~program_mode env sigma frozen false
    else sigma
  in
  let sigma = match hook with
  | None -> sigma
  | Some hook -> apply_inference_hook hook env sigma frozen
  in
  let sigma = if flags.solve_unification_constraints
    then apply_heuristics env sigma false
    else sigma
  in
  if flags.fail_evar then check_evars_are_solved ~program_mode env sigma frozen;
  sigma

let check_evars_are_solved ~program_mode env ?initial current_sigma =
  let frozen = frozen_and_pending_holes (initial, current_sigma) in
  check_evars_are_solved ~program_mode env current_sigma frozen

let process_inference_flags flags env initial (sigma,c,cty) =
  let sigma = solve_remaining_evars flags env ~initial sigma in
  let c = if flags.expand_evars then nf_evar sigma c else c in
  sigma,c,cty

let adjust_evar_source sigma na c =
  match na, kind sigma c with
  | Name id, Evar (evk,args) ->
     let evi = Evd.find sigma evk in
     begin match evi.evar_source with
     | loc, Evar_kinds.QuestionMark {
         Evar_kinds.qm_obligation=b;
         Evar_kinds.qm_name=Anonymous;
         Evar_kinds.qm_record_field=recfieldname;
        } ->
        let src = (loc,Evar_kinds.QuestionMark {
            Evar_kinds.qm_obligation=b;
            Evar_kinds.qm_name=na;
            Evar_kinds.qm_record_field=recfieldname;
        }) in
        let (sigma, evk') = restrict_evar sigma evk (evar_filter evi) ~src None in
        sigma, mkEvar (evk',args)
     | _ -> sigma, c
     end
  | _, _ -> sigma, c

(* coerce to tycon if any *)
let inh_conv_coerce_to_tycon ?loc ~program_mode resolve_tc env sigma j = function
  | None -> sigma, j, Some Coercion.empty_coercion_trace
  | Some t ->
    Coercion.inh_conv_coerce_to ?loc ~program_mode resolve_tc !!env sigma j t

let check_instance loc subst = function
  | [] -> ()
  | (id,_) :: _ ->
      if List.mem_assoc id subst then
        user_err ?loc  (Id.print id ++ str "appears more than once.")
      else
        user_err ?loc  (str "No such variable in the signature of the existential variable: " ++ Id.print id ++ str ".")

(* used to enforce a name in Lambda when the type constraints itself
   is named, hence possibly dependent *)

let orelse_name name name' = match name with
  | Anonymous -> name'
  | _ -> name

let inductive_to_string i =
  let _, (i : Declarations.one_inductive_body) = Global.lookup_inductive i in
  Id.to_string i.mind_typename

let constructor_to_string (ind, x) =
  let _, (ind : Declarations.one_inductive_body) = Global.lookup_inductive ind in
  let constr = ind.mind_consnames.(x - 1) in
  Id.to_string constr

let projection_to_string p =
  Label.to_string (Projection.label p)

let print_universe_instance sigma u =
  if not (EInstance.is_empty u) then
    let inst = EInstance.kind sigma u in
    let buf = Buffer.create 16 in
    Buffer.add_string buf "(";
    Array.iter (fun u ->
      if Buffer.length buf > 1 then Buffer.add_string buf " ";
      Buffer.add_string buf (Pp.string_of_ppcmds (Level.pr u))
    ) (Instance.to_array inst);
    Buffer.add_string buf ")";
    Buffer.contents buf
  else ""

open Context.Rel

let print_rel_context (ctx : Constr.rel_context) =
  let i = ref 1 in
  List.iter (fun decl ->
    let name =
      match decl with
      | Context.Rel.Declaration.LocalAssum (na, _)
      | Context.Rel.Declaration.LocalDef (na, _, _) ->
        match na.Context.binder_name with
        | Names.Name id -> Names.Id.to_string id
        | Names.Anonymous -> "_"
    in
    Printf.printf "  [%d]: %s (Rel %d)\n" !i name !i;
    incr i
  ) ctx

let print_named_context named_ctx =
  List.iteri (fun i decl ->
    let id = Context.Named.Declaration.get_id decl in
    Printf.printf "Named context [%d]: %s\n" (i + 1) (Id.to_string id)
  ) (List.rev named_ctx)
;;

let rec print_type env sigma id typ =
  let typ_constr = EConstr.to_constr sigma typ in
  let initial_ctx =
    let rel_ctx = Environ.rel_context env in
    let rec find_pos ctx i =
      match ctx with
      | [] -> None
      | decl::rest ->
          match Context.Rel.Declaration.get_name decl with
          | Names.Name id' when Id.equal id' id -> Some i
          | _ -> find_pos rest (i+1)
    in
    let trimmed_ctx =
      match find_pos rel_ctx 0 with
      | None -> rel_ctx
      | Some pos ->
          (* Printf.printf "Initial trimming at position %d for id %s\n" pos (Id.to_string id); *)
          List.skipn (pos+1) rel_ctx
    in
    List.map (fun decl ->
      match decl with
      | Context.Rel.Declaration.LocalAssum(na, ty) ->
          Context.Rel.Declaration.LocalAssum(na, EConstr.of_constr ty)
      | Context.Rel.Declaration.LocalDef(na, term, ty) ->
          Context.Rel.Declaration.LocalDef(na, EConstr.of_constr term, EConstr.of_constr ty)
    ) trimmed_ctx
  in
  let rec print_type_rec env ctx typ =
    match EConstr.kind sigma typ with
    | Rel n ->
      begin
        try
          let decl = List.nth ctx (n-1) in
          match Context.Rel.Declaration.get_name decl with
          | Names.Name id -> Id.to_string id
          | Names.Anonymous -> "Rel(" ^ string_of_int n ^ ")"
        with Failure _ | Invalid_argument _ ->
          "Rel(" ^ string_of_int n ^ ")"
      end
    | App (f, args) ->
      let f_str = print_type_rec env ctx f in
      let args_str = Array.fold_right (fun arg acc ->
        let arg_str = print_type_rec env ctx arg in
        arg_str ^ (if acc <> "" then " " ^ acc else "")
      ) args "" in
      "( " ^ f_str ^ " " ^ args_str ^ " )"
    | Ind ((mind, i), u) ->
      let mind_str = MutInd.to_string mind in
      let ind_str = inductive_to_string (mind, i) in
      mind_str ^ "." ^ ind_str
    | Lambda (na, t1, t2) ->
      let name = match Context.binder_name na with
        | Names.Name id -> Id.to_string id
        | Names.Anonymous -> "_Anonymous"
      in
      let t1_str = print_type_rec env ctx t1 in
      let new_ctx = Context.Rel.Declaration.LocalAssum(na, t1) :: ctx in
      let t2_str = print_type_rec env new_ctx t2 in
      "fun ( " ^ name ^ " : " ^ t1_str ^ " ) => " ^ t2_str
    | Case (ci, p, c, brs) ->
      let c_str = print_type_rec env ctx c in
      let p_str = print_type_rec env ctx p in
      let brs_str = Array.fold_left (fun acc br ->
        let br_str = print_type_rec env ctx br in
        acc ^ (if acc <> "" then " | " else "") ^ br_str
      ) "" brs in
      "match " ^ c_str ^ " with " ^ brs_str ^ " end"
    | Fix ((vn,i), (nams, tys, bds)) ->
      let fix_name = match Context.binder_name (Array.get nams i) with
        | Names.Name id -> Id.to_string id
        | Names.Anonymous -> "_Anonymous"
      in
      let ty_str = print_type_rec env ctx (Array.get tys i) in
      let body_str = print_type_rec env ctx (Array.get bds i) in
      " TYPE : " ^ ty_str ^ " BODY : " ^ body_str
    | CoFix (i, (nams, tys, bds)) ->
      let cofix_name = match Context.binder_name (Array.get nams i) with
        | Names.Name id -> Id.to_string id
        | Names.Anonymous -> "_Anonymous"
      in
      let ty_str = print_type_rec env ctx (Array.get tys i) in
      let body_str = print_type_rec env ctx (Array.get bds i) in
      " TYPE : " ^ ty_str ^ " BODY : " ^ body_str
    | Construct (((mind, i), j), u) ->
      let ind_name = MutInd.to_string mind in
      let actual_name = constructor_to_string ((mind, i), j) in
      let full_name = ind_name ^ "." ^ actual_name in
      full_name
    | Proj (p, c') ->
      let proj_name = Constant.debug_to_string_type (Projection.constant p) in
      let c_str = print_type_rec env ctx c' in
      proj_name ^ "." ^ c_str
    | Const (c, u) ->
      let const_name = Constant.debug_to_string_type c in
      const_name
    | Var id ->
      Id.to_string id
    | Evar (ev, cl) ->
      let evi = Evd.find sigma ev in
      print_type_rec env ctx evi.evar_concl
    | Int i ->
      Uint63.to_string i
    | Float f ->
      Float64.to_string f
    | Prod (na, t1, t2) ->
      let name = match Context.binder_name na with
        | Names.Name id -> Id.to_string id
        | Names.Anonymous -> "_Anonymous"
      in
      let t1_str = print_type_rec env ctx t1 in
      let new_ctx = Context.Rel.Declaration.LocalAssum(na, t1) :: ctx in
      let t2_str = print_type_rec env new_ctx t2 in
      "forall ( " ^ name ^ " : " ^ t1_str ^ " ) -> " ^ t2_str
      (* "forall " ^ name ^ " : " ^ t1_str ^ " -> " ^ t2_str *)
    | LetIn (na, t1, ty, t2) ->
      let name = match Context.binder_name na with
        | Names.Name id -> Id.to_string id
        | Names.Anonymous -> "_Anonymous"
      in
      let t1_str = print_type_rec env ctx t1 in
      let ty_str = print_type_rec env ctx ty in
      let new_ctx = Context.Rel.Declaration.LocalDef(na, t1, ty) :: ctx in
      let t2_str = print_type_rec env new_ctx t2 in
      "let " ^ name ^ " : " ^ ty_str ^ " := " ^ t1_str ^ " in " ^ t2_str
    | Cast (term, kind, typ) ->
      let term_str = print_type_rec env ctx term in
      let typ_str = print_type_rec env ctx typ in
      "( " ^ term_str ^ " : " ^ typ_str ^ " )"
    | Sort s ->
      begin match ESorts.kind sigma s with
      | Sorts.SProp -> "SProp"
      | Sorts.Prop -> "Prop"
      | Sorts.Set -> "Set"
      | Sorts.Type u -> "Type"
      end
    | _ ->
      (* The Meta variable case might not be essential *)
      Pp.string_of_ppcmds (debug_print typ_constr)
  in
  print_type_rec env initial_ctx typ

let compute_displayed_name_in_pattern sigma avoid na c =
  let open Namegen in
  compute_displayed_name_in_gen (fun _ -> Patternops.noccurn_pattern) sigma avoid na c

let any_any_branch =
  (* | _ => _ *)
  CAst.make ([],[DAst.make @@ PatVar Anonymous], DAst.make @@ GHole (Evar_kinds.InternalHole,IntroAnonymous,None))

open Pattern

let rec glob_of_pat avoid env sigma pat = DAst.make @@ match pat with
| PRef ref -> GRef (ref,None)
| PVar id  -> GVar id
| PEvar (evk,l) ->
    let test decl = function PVar id' -> Id.equal (NamedDecl.get_id decl) id' | _ -> false in
    let l = Evd.evar_instance_array test (Evd.find sigma evk) l in
    let id = match Evd.evar_ident evk sigma with
    | None -> Id.of_string "__"
    | Some id -> id
    in
    GEvar (id,List.map (on_snd (glob_of_pat avoid env sigma)) l)
| PRel n ->
    let id = try match lookup_name_of_rel n env with
      | Name id   ->
        (* Printf.printf "Rel(%d): %s\n" n (Id.to_string id); *)
        id
      | Anonymous ->
          (* Printf.printf "Rel(%d): Rel(%d)\n" n n; *)
          Id.of_string ("_ANONYMOUS_REL_"^(string_of_int n))
    with Not_found -> Id.of_string ("_UNBOUND_REL_"^(string_of_int n)) in
    GVar id
| PMeta None -> GHole (Evar_kinds.InternalHole, IntroAnonymous,None)
| PMeta (Some n) -> GPatVar (Evar_kinds.FirstOrderPatVar n)
| PProj (p,c) -> GApp (DAst.make @@ GRef (GlobRef.ConstRef (Projection.constant p),None),
                        [glob_of_pat avoid env sigma c])
| PApp (f,args) ->
    GApp (glob_of_pat avoid env sigma f,Array.map_to_list (glob_of_pat avoid env sigma) args)
| PSoApp (n,args) ->
    GApp (DAst.make @@ GPatVar (Evar_kinds.SecondOrderPatVar n),
      List.map (glob_of_pat avoid env sigma) args)
| PProd (na,t,c) ->
    let na',avoid' = compute_displayed_name_in_pattern sigma avoid na c in
    let env' = Termops.add_name na' env in
    GProd (na',Explicit,glob_of_pat avoid env sigma t,glob_of_pat avoid' env' sigma c)
| PLetIn (na,b,t,c) ->
    let na',avoid' = Namegen.compute_displayed_let_name_in sigma Namegen.RenamingForGoal avoid na c in
    let env' = Termops.add_name na' env in
    GLetIn (na',glob_of_pat avoid env sigma b, Option.map (glob_of_pat avoid env sigma) t,
            glob_of_pat avoid' env' sigma c)
| PLambda (na,t,c) ->
    let na',avoid' = compute_displayed_name_in_pattern sigma avoid na c in
    let env' = Termops.add_name na' env in
    GLambda (na',Explicit,glob_of_pat avoid env sigma t, glob_of_pat avoid' env' sigma c)
| PIf (c,b1,b2) ->
    GIf (glob_of_pat avoid env sigma c, (Anonymous,None),
          glob_of_pat avoid env sigma b1, glob_of_pat avoid env sigma b2)
| PCase ({cip_style=Constr.LetStyle; cip_ind_tags=None},PMeta None,tm,[(0,n,b)]) ->
    let nal,b = Detyping.it_destRLambda_or_LetIn_names n (glob_of_pat avoid env sigma b) in
    GLetTuple (nal,(Anonymous,None),glob_of_pat avoid env sigma tm,b)
| PCase (info,p,tm,bl) ->
    let mat = match bl, info.cip_ind with
      | [], _ -> []
      | _, Some ind ->
        let bl' = List.map (fun (i,n,c) -> (i,n,glob_of_pat avoid env sigma c)) bl in
        Detyping.simple_cases_matrix_of_branches ind bl'
      | _, None -> anomaly (Pp.str "PCase with some branches but unknown inductive.")
    in
    let mat = if info.cip_extensible then mat @ [any_any_branch] else mat
    in
    let indnames,rtn = match p, info.cip_ind, info.cip_ind_tags with
      | PMeta None, _, _ -> (Anonymous,None),None
      | _, Some ind, Some nargs ->
        Detyping.return_type_of_predicate ind nargs (glob_of_pat avoid env sigma p)
      | _ -> anomaly (Pp.str "PCase with non-trivial predicate but unknown inductive.")
    in
    GCases (Constr.RegularStyle,rtn,[glob_of_pat avoid env sigma tm,indnames],mat)
| PFix ((ln,i),(lna,tl,bl)) ->
    let def_avoid, def_env, lfi =
      Array.fold_left
        (fun (avoid, env, l) na ->
          let id = Namegen.next_name_away na avoid in
          (Id.Set.add id avoid, Name id :: env, id::l))
    (avoid, env, []) lna in
    let n = Array.length tl in
    let v = Array.map3
              (fun c t i -> Detyping.share_pattern_names glob_of_pat (i+1) [] def_avoid def_env sigma c (Patternops.lift_pattern n t))
  bl tl ln in
    GRec(GFix (Array.map (fun i -> Some i) ln,i),Array.of_list (List.rev lfi),
      Array.map (fun (bl,_,_) -> bl) v,
      Array.map (fun (_,_,ty) -> ty) v,
      Array.map (fun (_,bd,_) -> bd) v)
| PCoFix (ln,(lna,tl,bl)) ->
    let def_avoid, def_env, lfi =
      Array.fold_left
        (fun (avoid, env, l) na ->
          let id = Namegen.next_name_away na avoid in
          (Id.Set.add id avoid, Name id :: env, id::l))
        (avoid, env, []) lna in
    let ntys = Array.length tl in
    let v = Array.map2
              (fun c t -> Detyping.share_pattern_names glob_of_pat 0 [] def_avoid def_env sigma c (Patternops.lift_pattern ntys t))
              bl tl in
    GRec(GCoFix ln,Array.of_list (List.rev lfi),
        Array.map (fun (bl,_,_) -> bl) v,
        Array.map (fun (_,_,ty) -> ty) v,
        Array.map (fun (_,bd,_) -> bd) v)
| PSort Sorts.InSProp -> GSort (UNamed [GSProp,0])
| PSort Sorts.InProp -> GSort (UNamed [GProp,0])
| PSort Sorts.InSet -> GSort (UNamed [GSet,0])
| PSort Sorts.InType -> GSort (UAnonymous {rigid=true})
| PInt i -> GInt i
| PFloat f -> GFloat f

let print_var_info env sigma id typ =
  Printf.printf "Variable %s : %s\n" (Id.to_string id) (print_type env sigma id typ);
  (* let typ_constr = EConstr.to_constr sigma typ in *)
  (* let pat = Patternops.pattern_of_constr env sigma typ_constr in *)
  (* let _ = glob_of_pat Id.Set.empty (Termops.names_of_rel_context env) sigma pat in  *)
  (* () *)


open Format
let std_ft = ref Format.std_formatter
let pp   x = Pp.pp_with !std_ft x

let cast_kind_display k =
  match k with
  | VMcast -> "VMcast"
  | DEFAULTcast -> "DEFAULTcast"
  | REVERTcast -> "REVERTcast"
  | NATIVEcast -> "NATIVEcast"

let cnt = ref 0

let print_constr env sigma ct =
  let csr = EConstr.to_constr sigma ct in
  let rec term_display c = match Constr.kind c with
  | Rel n -> "Rel("^(string_of_int n)^")"
  | Meta n -> "Meta("^(string_of_int n)^")"
  | Var id -> "Var("^(Id.to_string id)^")"
  | Sort s -> "Sort("^(sort_display s)^")"
  | Cast (c,k, t) ->
      "Cast("^(term_display c)^","^(cast_kind_display k)^","^(term_display t)^")"
  | Prod (na,t,c) ->
      "Prod("^(name_display na)^","^(term_display t)^","^(term_display c)^")"
  | Lambda (na,t,c) ->
      "Lambda("^(name_display na)^","^(term_display t)^","^(term_display c)^")"
  | LetIn (na,b,t,c) ->
      "LetIn("^(name_display na)^","^(term_display b)^","
      ^(term_display t)^","^(term_display c)^")"
  | App (c,l) -> "App("^(term_display c)^","^(array_display l)^")"
  | Evar (e,l) -> "Evar("^(Pp.string_of_ppcmds (Evar.print e))^","^(array_display l)^")"
  | Const (c,u) -> "Const("^(Constant.to_string c)^","^(universes_display u)^")"
  | Ind ((sp,i),u) ->
      "MutInd("^(MutInd.to_string sp)^","^(string_of_int i)^","^(universes_display u)^")"
  | Construct (((sp,i),j),u) ->
      "MutConstruct(("^(MutInd.to_string sp)^","^(string_of_int i)^"),"
      ^","^(universes_display u)^(string_of_int j)^")"
  | Proj (p, c) -> "Proj("^(Constant.to_string (Projection.constant p))^","^term_display c ^")"
  | Case (ci,p,c,bl) ->
      "MutCase(<abs>,"^(term_display p)^","^(term_display c)^","
      ^(array_display bl)^")"
  | Fix ((t,i),(lna,tl,bl)) ->
      "Fix(([|"^(Array.fold_right (fun x i -> (string_of_int x)^(if not(i="")
        then (";"^i) else "")) t "")^"|],"^(string_of_int i)^"),"
      ^(array_display tl)^",[|"
      ^(Array.fold_right (fun x i -> (name_display x)^(if not(i="")
        then (";"^i) else "")) lna "")^"|],"
      ^(array_display bl)^")"
  | CoFix(i,(lna,tl,bl)) ->
      "CoFix("^(string_of_int i)^"),"
      ^(array_display tl)^","
      ^(Array.fold_right (fun x i -> (name_display x)^(if not(i="")
        then (";"^i) else "")) lna "")^","
      ^(array_display bl)^")"
  | Int i ->
      "Int("^(Uint63.to_string i)^")"
  | Float f ->
      "Float("^(Float64.to_string f)^")"

  and array_display v =
    "[|"^
    (Array.fold_right
       (fun x i -> (term_display x)^(if not(i="") then (";"^i) else ""))
       v "")^"|]"

  and univ_display u =
    incr cnt; pp (str "with " ++ int !cnt ++ str" " ++ pr_uni u ++ fnl ())

  and level_display u =
    incr cnt; pp (str "with " ++ int !cnt ++ str" " ++ Level.pr u ++ fnl ())

  and sort_display = function
    | SProp -> "SProp"
    | Set -> "Set"
    | Prop -> "Prop"
    (* | Type u -> univ_display u;
        "Type("^(string_of_int !cnt)^")" *)
    | Type u -> univ_display u;"Type("^(string_of_int !cnt)^")"

  and universes_display l =
    Array.fold_right (fun x i -> level_display x; (string_of_int !cnt)^(if not(i="")
        then (" "^i) else "")) (Instance.to_array l) ""

  and name_display x = match x.binder_name with
    | Name id -> "Name("^(Id.to_string id)^")"
    | Anonymous -> "_Anonymous"

  in
  pp (str (term_display csr) ++ fnl ());
  Format.pp_print_flush !std_ft ()

let print_pure_constr env sigma ct =
  let csr = EConstr.to_constr sigma ct in
  let rec term_display c = match Constr.kind c with
  | Rel n -> print_string "#";
    (* let rel_ctx = Environ.rel_context env in
    Printf.printf "\nLooking for Rel(%d)\n" n;
    Printf.printf "Handling Rel(%d)\n" n;
    Printf.printf "Current Rel Context:\n";
    print_rel_context rel_ctx;
    print_int n *)
  | Meta n -> print_string "Meta("; print_int n; print_string ")"
  | Var id -> print_string (Id.to_string id)
  | Sort s -> sort_display s
  | Cast (c,_, t) -> open_hovbox 1;
      print_string "("; (term_display c); print_cut();
      print_string "::"; (term_display t); print_string ")"; close_box()
  | Prod ({binder_name=Name(id)},t,c) ->
      open_hovbox 1;
      print_string"("; print_string (Id.to_string id);
      print_string ":"; box_display t;
      print_string ")"; print_cut();
      box_display c; close_box()
  | Prod ({binder_name=Anonymous},t,c) ->
      print_string"("; box_display t; print_cut(); print_string "->";
      box_display c; print_string ")";
  | Lambda (na,t,c) ->
      print_string "["; name_display na;
      print_string ":"; box_display t; print_string "]";
      print_cut(); box_display c;
  | LetIn (na,b,t,c) ->
      print_string "["; name_display na; print_string "=";
      box_display b; print_cut();
      print_string ":"; box_display t; print_string "]";
      print_cut(); box_display c;
  | App (c,l) ->
      print_string "(";
      box_display c;
      Array.iter (fun x -> print_space (); box_display x) l;
      print_string ")"
  | Evar (e,l) -> print_string "Evar#"; print_int (Evar.repr e); print_string "{";
      Array.iter (fun x -> print_space (); box_display x) l;
      print_string"}"
  | Const (c,u) -> print_string "Cons(";
      sp_con_display c;
      print_string ","; universes_display u;
      print_string ")"
  | Proj (p,c') -> print_string "Proj(";
      sp_con_display (Projection.constant p);
      print_string ",";
      box_display c';
      print_string ")"
  | Ind ((sp,i),u) ->
      print_string "Ind(";
      sp_display sp;
      print_string ","; print_int i;
      print_string ","; universes_display u;
      print_string ")"
  | Construct (((sp,i),j),u) ->
      print_string "Constr(";
      sp_display sp;
      print_string ",";
      print_int i; print_string ","; print_int j;
      print_string ","; universes_display u;
      print_string ")"
  | Case (ci,p,c,bl) ->
      open_vbox 0;
      print_string "<"; box_display p; print_string ">";
      print_cut(); print_string "Case";
      print_space(); box_display c; print_space (); print_string "of";
      open_vbox 0;
      Array.iter (fun x ->  print_cut();  box_display x) bl;
      close_box();
      print_cut();
      print_string "end";
      close_box()
  | Fix ((t,i),(lna,tl,bl)) ->
      print_string "Fix("; print_int i; print_string ")";
      print_cut();
      open_vbox 0;
      let print_fix () =
        for k = 0 to (Array.length tl) - 1 do
          open_vbox 0;
          name_display lna.(k); print_string "/";
          print_int t.(k); print_cut(); print_string ":";
          box_display tl.(k) ; print_cut(); print_string ":=";
          box_display bl.(k); close_box ();
          print_cut()
        done
      in print_string"{"; print_fix(); print_string"}"
  | CoFix(i,(lna,tl,bl)) ->
      print_string "CoFix("; print_int i; print_string ")";
      print_cut();
      open_vbox 0;
      let print_fix () =
        for k = 0 to (Array.length tl) - 1 do
          open_vbox 1;
          name_display lna.(k);  print_cut(); print_string ":";
          box_display tl.(k) ; print_cut(); print_string ":=";
          box_display bl.(k); close_box ();
          print_cut();
        done
      in print_string"{"; print_fix (); print_string"}"
  | Int i ->
      print_string ("Int("^(Uint63.to_string i)^")")
  | Float f ->
      print_string ("Float("^(Float64.to_string f)^")")

  and box_display c = open_hovbox 1; term_display c; close_box()

  and universes_display u =
    Array.iter (fun u -> print_space (); pp (Level.pr u)) (Instance.to_array u)

  and sort_display = function
    | SProp -> print_string "SProp"
    | Set -> print_string "Set"
    | Prop -> print_string "Prop"
    | Type u -> open_hbox();
        print_string "Type("; pp (pr_uni u); print_string ")"; close_box()

  and name_display x = match x.binder_name with
    | Name id -> print_string (Id.to_string id)
    | Anonymous -> print_string "_Anonymous"
(* Remove the top names for library and Scratch to avoid long names *)
  and sp_display sp =
(*    let dir,l = decode_kn sp in
    let ls =
      match List.rev_map Id.to_string (DirPath.repr dir) with
          ("Top"::l)-> l
        | ("Coq"::_::l) -> l
        | l             -> l
    in  List.iter (fun x -> print_string x; print_string ".") ls;*)
      print_string (MutInd.debug_to_string sp)
  and sp_con_display sp =
(*    let dir,l = decode_kn sp in
    let ls =
      match List.rev_map Id.to_string (DirPath.repr dir) with
          ("Top"::l)-> l
        | ("Coq"::_::l) -> l
        | l             -> l
    in  List.iter (fun x -> print_string x; print_string ".") ls;*)
      print_string (Constant.debug_to_string sp)

  in
    try
      box_display csr; print_flush()
    with e ->
        print_string (Printexc.to_string e);print_flush ();
        raise e
    Format.pp_print_flush !std_ft ()

let print_type_func env sigma id typ =
  (* () *)
  Printf.printf "pure constr: ";
  print_pure_constr env sigma typ;
  print_newline ();
  Printf.printf "constr: ";
  print_constr env sigma typ;
  (* print_newline (); *)
  Printf.printf "processed: ";
  print_var_info env sigma id typ;
  print_newline ()

let print_type_func_with_keyword env sigma id typ keyword =
  let keyword = Some keyword in
  (match keyword with
    | Some k -> Printf.printf "======= Begin %s ========\n" k
    | None -> ());

  Printf.printf "pure constr: ";
  print_pure_constr env sigma typ;
  print_newline ();
  Printf.printf "constr: ";
  print_constr env sigma typ;
  Printf.printf "processed: ";
  print_var_info env sigma id typ;
  print_newline ();

  (match keyword with
    | Some k -> Printf.printf "======= End %s ========\n" k
    | None -> ())

let print_constr_title x =
  Printf.printf "Constr Begin %s\n\n" x

let print_constr_end x =
  Printf.printf "Constr End %s\n\n" x

let pretype_id pretype loc env sigma id =
  (* Look for the binder of [id] *)
  try
    let (n,_,typ) = lookup_rel_id id (rel_context !!env) in
    (* Printf.printf "11111111"; *)
    (* Printf.printf "Type of variable %s: " (Id.to_string id); *)
    Printf.printf "Type Begin\n";
    print_type_func !!env sigma id typ;
    Printf.printf "Type End\n\n";
    (* Printf.printf "222222222"; *)
    sigma, { uj_val  = mkRel n; uj_type = lift n typ }
  with Not_found ->
  try
    (* Printf.printf "11111111"; *)
    GlobEnv.interp_ltac_variable ?loc (fun env -> pretype env sigma) env sigma id
  with Not_found ->
  (* Check if [id] is a section or goal variable *)
  try
    let typ = NamedDecl.get_type (lookup_named id !!env) in
    (* Printf.printf "Type of variable %s: " (Id.to_string id); *)
    Printf.printf "Type Begin\n";
    print_type_func !!env sigma id typ;
    Printf.printf "Type End\n\n";
    sigma, { uj_val  = mkVar id; uj_type = typ }
  with Not_found ->
    (* Printf.printf "333333333"; *)
    (* [id] not found, standard error message *)
    error_var_not_found ?loc !!env sigma id

(* Main pretyping function                                               *)

let interp_known_glob_level ?loc evd = function
  | GSProp -> Univ.Level.sprop
  | GProp -> Univ.Level.prop
  | GSet -> Univ.Level.set
  | GType qid ->
    try interp_known_universe_level_name evd qid
    with Not_found ->
      user_err ?loc ~hdr:"interp_known_level_info" (str "Undeclared universe " ++ Libnames.pr_qualid qid)

let interp_glob_level ?loc evd : glob_level -> _ = function
  | UAnonymous {rigid} -> new_univ_level_variable ?loc (if rigid then univ_rigid else univ_flexible) evd
  | UNamed s -> interp_sort_name ?loc evd s

let interp_instance ?loc evd l =
  let evd, l' =
    List.fold_left
      (fun (evd, univs) l ->
         let evd, l = interp_glob_level ?loc evd l in
         (evd, l :: univs)) (evd, [])
      l
  in
  if List.exists (fun l -> Univ.Level.is_prop l) l' then
    user_err ?loc ~hdr:"pretype"
      (str "Universe instances cannot contain Prop, polymorphic" ++
       str " universe instances must be greater or equal to Set.");
  evd, Some (Univ.Instance.of_array (Array.of_list (List.rev l')))

let pretype_global ?loc rigid env evd gr us =
  let evd, instance =
    match us with
    | None -> evd, None
    | Some l -> interp_instance ?loc evd l
  in
  Evd.fresh_global ?loc ~rigid ?names:instance !!env evd gr

let pretype_ref ?loc sigma env ref us =
  match ref with
  | GlobRef.VarRef id ->
      (* Section variable *)
      (* (try sigma, make_judge (mkVar id) (NamedDecl.get_type (lookup_named id !!env)) *)
      (try 
        let typ = NamedDecl.get_type (lookup_named id !!env) in
        Printf.printf "Type Begin\n";
        print_type_func !!env sigma id typ;
        Printf.printf "Type End\n\n";
        sigma, make_judge (mkVar id) typ
       with Not_found ->
         (* This may happen if env is a goal env and section variables have
            been cleared - section variables should be different from goal
            variables *)
         Pretype_errors.error_var_not_found ?loc !!env sigma id)
  | ref ->
    let sigma, c = pretype_global ?loc univ_flexible env sigma ref us in
    let ty = unsafe_type_of !!env sigma c in
    (match ref with
    | GlobRef.ConstRef const ->
        Printf.printf "Type Begin\n";
        let full_name = Constant.to_string const in
        let parts = String.split_on_char '.' full_name in
        let last_part = List.nth parts (List.length parts - 1) in
        print_type_func !!env sigma (Id.of_string last_part) ty;
        Printf.printf "Type End\n\n"
    | _ -> ());
   sigma, make_judge c ty

let interp_sort ?loc evd : glob_sort -> _ = function
  | UAnonymous {rigid} ->
    let evd, l = new_univ_level_variable ?loc (if rigid then univ_rigid else univ_flexible) evd in
    evd, Univ.Universe.make l
  | UNamed l -> interp_sort_info ?loc evd l

let judge_of_sort ?loc evd s =
  let judge =
    { uj_val = mkType s; uj_type = mkType (Univ.super s) }
  in
    evd, judge

let pretype_sort ?loc sigma s =
  match s with
  | UNamed [GSProp,0] -> sigma, judge_of_sprop
  | UNamed [GProp,0] -> sigma, judge_of_prop
  | UNamed [GSet,0] -> sigma, judge_of_set
  | _ ->
  let sigma, s = interp_sort ?loc sigma s in
  judge_of_sort ?loc sigma s

let new_type_evar env sigma loc =
  new_type_evar env sigma ~src:(Loc.tag ?loc Evar_kinds.InternalHole)

let mark_obligation_evar sigma k evc =
  match k with
  | Evar_kinds.QuestionMark _
  | Evar_kinds.ImplicitArg (_, _, false) ->
    Evd.set_obligation_evar sigma (fst (destEvar sigma evc))
  | _ -> sigma

(* [pretype tycon env sigma lvar lmeta cstr] attempts to type [cstr] *)
(* in environment [env], with existential variables [sigma] and *)
(* the type constraint tycon *)

let discard_trace (sigma,t,otrace) = sigma, t

let rec pretype ~program_mode ~poly resolve_tc (tycon : type_constraint) (env : GlobEnv.t) (sigma : evar_map) t =
  let inh_conv_coerce_to_tycon ?loc = inh_conv_coerce_to_tycon ?loc ~program_mode resolve_tc in
  let pretype_type = pretype_type ~program_mode ~poly resolve_tc in
  let pretype = pretype ~program_mode ~poly resolve_tc in
  let open Context.Rel.Declaration in
  let loc = t.CAst.loc in
  match DAst.get t with
  | GRef (ref,u) ->
    let sigma, t_ref = pretype_ref ?loc sigma env ref u in
    discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma t_ref tycon

  | GVar id ->
    let sigma, t_id = pretype_id (fun e r t -> pretype tycon e r t) loc env sigma id in
    discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma t_id tycon

  | GEvar (id, inst) ->
      (* Ne faudrait-il pas s'assurer que hyps est bien un
         sous-contexte du contexte courant, et qu'il n'y a pas de Rel "caché" *)
      let id = interp_ltac_id env id in
      let evk =
        try Evd.evar_key id sigma
        with Not_found -> error_evar_not_found ?loc !!env sigma id in
      let hyps = evar_filtered_context (Evd.find sigma evk) in
      let sigma, args = pretype_instance ~program_mode ~poly resolve_tc env sigma loc hyps evk inst in
      let c = mkEvar (evk, args) in
      let j = Retyping.get_judgment_of !!env sigma c in
      discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma j tycon

  | GPatVar kind ->
    let sigma, ty =
      match tycon with
      | Some ty -> sigma, ty
      | None -> new_type_evar env sigma loc in
    let k = Evar_kinds.MatchingVar kind in
    let sigma, uj_val = new_evar env sigma ~src:(loc,k) ty in
    sigma, { uj_val; uj_type = ty }

  | GHole (k, naming, None) ->
      let open Namegen in
      let naming = match naming with
        | IntroIdentifier id -> IntroIdentifier (interp_ltac_id env id)
        | IntroAnonymous -> IntroAnonymous
        | IntroFresh id -> IntroFresh (interp_ltac_id env id) in
      let sigma, ty =
        match tycon with
        | Some ty -> sigma, ty
        | None -> new_type_evar env sigma loc in
      let sigma, uj_val = new_evar env sigma ~src:(loc,k) ~naming ty in
      let sigma = if program_mode then mark_obligation_evar sigma k uj_val else sigma in
      sigma, { uj_val; uj_type = ty }

  | GHole (k, _naming, Some arg) ->
      let sigma, ty =
        match tycon with
        | Some ty -> sigma, ty
        | None -> new_type_evar env sigma loc in
      let c, sigma = GlobEnv.interp_glob_genarg env poly sigma ty arg in
      sigma, { uj_val = c; uj_type = ty }

  | GRec (fixkind,names,bl,lar,vdef) ->
    let hypnaming = if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames in
    let rec type_bl env sigma ctxt = function
      | [] -> sigma, ctxt
      | (na,bk,None,ty)::bl ->
        let sigma, ty' = pretype_type empty_valcon env sigma ty in
        let rty' = Sorts.relevance_of_sort ty'.utj_type in
        let dcl = LocalAssum (make_annot na rty', ty'.utj_val) in
        let dcl', env = push_rel ~hypnaming sigma dcl env in
        type_bl env sigma (Context.Rel.add dcl' ctxt) bl
      | (na,bk,Some bd,ty)::bl ->
        let sigma, ty' = pretype_type empty_valcon env sigma ty in
        let rty' = Sorts.relevance_of_sort ty'.utj_type in
        let sigma, bd' = pretype (mk_tycon ty'.utj_val) env sigma bd in
        let dcl = LocalDef (make_annot na rty', bd'.uj_val, ty'.utj_val) in
        let dcl', env = push_rel ~hypnaming sigma dcl env in
        type_bl env sigma (Context.Rel.add dcl' ctxt) bl in
    let sigma, ctxtv = Array.fold_left_map (fun sigma -> type_bl env sigma Context.Rel.empty) sigma bl in
    let sigma, larj =
      Array.fold_left2_map
        (fun sigma e ar ->
          pretype_type empty_valcon (snd (push_rel_context ~hypnaming sigma e env)) sigma ar)
        sigma ctxtv lar in
    let lara = Array.map (fun a -> a.utj_val) larj in
    let ftys = Array.map2 (fun e a -> it_mkProd_or_LetIn a e) ctxtv lara in
    let nbfix = Array.length lar in
    let names = Array.map (fun id -> Name id) names in
    let sigma =
      match tycon with
      | Some t ->
        let fixi = match fixkind with
          | GFix (vn,i) -> i
          | GCoFix i -> i
        in
        begin match Evarconv.unify_delay !!env sigma ftys.(fixi) t with
          | exception Evarconv.UnableToUnify _ -> sigma
          | sigma -> sigma
        end
      | None -> sigma
    in
    let names = Array.map2 (fun na t ->
        make_annot na (Retyping.relevance_of_type !!(env) sigma t))
        names ftys
    in
      (* Note: bodies are not used by push_rec_types, so [||] is safe *)
    let names,newenv = push_rec_types ~hypnaming sigma (names,ftys) env in
    let sigma, vdefj =
      Array.fold_left2_map_i
        (fun i sigma ctxt def ->
           (* we lift nbfix times the type in tycon, because of
            * the nbfix variables pushed to newenv *)
           let (ctxt,ty) =
             decompose_prod_n_assum sigma (Context.Rel.length ctxt)
               (lift nbfix ftys.(i)) in
           let ctxt,nenv = push_rel_context ~hypnaming sigma ctxt newenv in
           let sigma, j = pretype (mk_tycon ty) nenv sigma def in
           sigma, { uj_val = it_mkLambda_or_LetIn j.uj_val ctxt;
                    uj_type = it_mkProd_or_LetIn j.uj_type ctxt })
        sigma ctxtv vdef in
      let sigma = Typing.check_type_fixpoint ?loc !!env sigma names ftys vdefj in
      let nf c = nf_evar sigma c in
      let ftys = Array.map nf ftys in (* FIXME *)
      let fdefs = Array.map (fun x -> nf (j_val x)) vdefj in
      let fixj = match fixkind with
        | GFix (vn,i) ->
              (* First, let's find the guard indexes. *)
              (* If recursive argument was not given by user, we try all args.
                 An earlier approach was to look only for inductive arguments,
                 but doing it properly involves delta-reduction, and it finally
                 doesn't seem worth the effort (except for huge mutual
                 fixpoints ?) *)
          let possible_indexes =
            Array.to_list (Array.mapi
                             (fun i annot -> match annot with
                             | Some n -> [n]
                             | None -> List.map_i (fun i _ -> i) 0 ctxtv.(i))
           vn)
          in
          let fixdecls = (names,ftys,fdefs) in
          let indexes = esearch_guard ?loc !!env sigma possible_indexes fixdecls in
          make_judge (mkFix ((indexes,i),fixdecls)) ftys.(i)
        | GCoFix i ->
          let fixdecls = (names,ftys,fdefs) in
          let cofix = (i, fixdecls) in
            (try check_cofix !!env (i, nf_fix sigma fixdecls)
             with reraise ->
               let (e, info) = CErrors.push reraise in
               let info = Option.cata (Loc.add_loc info) info loc in
               iraise (e, info));
            make_judge (mkCoFix cofix) ftys.(i)
      in
      discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma fixj tycon

  | GSort s ->
    let sigma, j = pretype_sort ?loc sigma s in
    discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma j tycon

  | GApp (f,args) ->
    let sigma, fj = pretype empty_tycon env sigma f in
    let floc = loc_of_glob_constr f in
    let length = List.length args in
    let nargs_before_bidi =
      if Option.is_empty tycon then length
      (* We apply bidirectionality hints only if an expected type is specified *)
      else
      (* if `f` is a global, we retrieve bidirectionality hints *)
        try
          let (gr,_) = destRef sigma fj.uj_val in
          Option.default length @@ GlobRef.Map.find_opt gr !bidi_hints
        with DestKO ->
          length
    in
    let candargs =
      (* Bidirectional typechecking hint:
         parameters of a constructor are completely determined
         by a typing constraint *)
      (* This bidirectionality machinery is the one of `Program` for
         constructors and is orthogonal to bidirectionality hints. However, we
         could probably factorize both by providing default bidirectionality hints
         for constructors corresponding to their number of parameters. *)
      if program_mode && length > 0 && isConstruct sigma fj.uj_val then
        match tycon with
        | None -> []
        | Some ty ->
          let ((ind, i), u) = destConstruct sigma fj.uj_val in
          let npars = inductive_nparams !!env ind in
          if Int.equal npars 0 then []
          else
            try
              let IndType (indf, args) = find_rectype !!env sigma ty in
              let ((ind',u'),pars) = dest_ind_family indf in
              if eq_ind ind ind' then List.map EConstr.of_constr pars
              else (* Let the usual code throw an error *) []
            with Not_found -> []
      else []
    in
    let app_f =
      match EConstr.kind sigma fj.uj_val with
      | Const (p, u) when Recordops.is_primitive_projection p ->
        let p = Option.get @@ Recordops.find_primitive_projection p in
        let p = Projection.make p false in
        let npars = Projection.npars p in
        fun n ->
          if Int.equal n npars then fun _ v -> mkProj (p, v)
          else fun f v -> applist (f, [v])
      | _ -> fun _ f v -> applist (f, [v])
    in
    let refresh_template env sigma resj =
      (* Special case for inductive type applications that must be
         refreshed right away. *)
      match EConstr.kind sigma resj.uj_val with
      | App (f,args) ->
        if Termops.is_template_polymorphic_ind !!env sigma f then
          let c = mkApp (f, args) in
          let sigma, c = Evarsolve.refresh_universes (Some true) !!env sigma c in
          let t = Retyping.get_type_of !!env sigma c in
          sigma, make_judge c (* use this for keeping evars: resj.uj_val *) t
        else sigma, resj
      | _ -> sigma, resj
    in
    let rec apply_rec env sigma n resj resj_before_bidi candargs bidiargs = function
      | [] -> sigma, resj, resj_before_bidi, List.rev bidiargs
      | c::rest ->
        let bidi = n >= nargs_before_bidi in
        let argloc = loc_of_glob_constr c in
        let sigma, resj, trace = Coercion.inh_app_fun ~program_mode resolve_tc !!env sigma resj in
        let resty = whd_all !!env sigma resj.uj_type in
        match EConstr.kind sigma resty with
        | Prod (na,c1,c2) ->
          let (sigma, hj), bidiargs =
            if bidi then
              (* We want to get some typing information from the context before
              typing the argument, so we replace it by an existential
              variable *)
              let sigma, c_hole = new_evar env sigma ~src:(loc,Evar_kinds.InternalHole) c1 in
              (sigma, make_judge c_hole c1), (c_hole, c, trace) :: bidiargs
            else
              let tycon = Some c1 in
              pretype tycon env sigma c, bidiargs
          in
          let sigma, candargs, ujval =
            match candargs with
            | [] -> sigma, [], j_val hj
            | arg :: args ->
              begin match Evarconv.unify_delay !!env sigma (j_val hj) arg with
                | exception Evarconv.UnableToUnify _ ->
                  sigma, [], j_val hj
                | sigma ->
                  sigma, args, nf_evar sigma (j_val hj)
              end
          in
          let sigma, ujval = adjust_evar_source sigma na.binder_name ujval in
          let value, typ = app_f n (j_val resj) ujval, subst1 ujval c2 in
          let resj = { uj_val = value; uj_type = typ } in
          let resj_before_bidi = if bidi then resj_before_bidi else resj in
          apply_rec env sigma (n+1) resj resj_before_bidi candargs bidiargs rest
        | _ ->
          let sigma, hj = pretype empty_tycon env sigma c in
          error_cant_apply_not_functional
            ?loc:(Loc.merge_opt floc argloc) !!env sigma resj [|hj|]
    in
    let sigma, resj, resj_before_bidi, bidiargs = apply_rec env sigma 0 fj fj candargs [] args in
    let sigma, resj = refresh_template env sigma resj in
    let sigma, resj, otrace = inh_conv_coerce_to_tycon ?loc env sigma resj tycon in
    let refine_arg n (sigma,t) (newarg,origarg,trace) =
      (* Refine an argument (originally `origarg`) represented by an evar
         (`newarg`) to use typing information from the context *)
      (* Recover the expected type of the argument *)
      let ty = Retyping.get_type_of !!env sigma newarg in
      (* Type the argument using this expected type *)
      let sigma, j = pretype (Some ty) env sigma origarg in
      (* Unify the (possibly refined) existential variable with the
      (typechecked) original value *)
      let sigma = Evarconv.unify_delay !!env sigma newarg (j_val j) in
      sigma, app_f n (Coercion.reapply_coercions sigma trace t) (j_val j)
    in
    (* We now refine any arguments whose typing was delayed for
       bidirectionality *)
    let t = resj_before_bidi.uj_val in
    let sigma, t = List.fold_left_i refine_arg nargs_before_bidi (sigma,t) bidiargs in
    (* If we did not get a coercion trace (e.g. with `Program` coercions, we
    replaced user-provided arguments with inferred ones. Otherwise, we apply
    the coercion trace to the user-provided arguments. *)
    let resj =
      match otrace with
      | None -> resj
      | Some trace ->
        let resj = { resj with uj_val = t } in
        let sigma, resj = refresh_template env sigma resj in
        { resj with uj_val = Coercion.reapply_coercions sigma trace t }
    in
    (sigma, resj)

  | GLambda(name,bk,c1,c2) ->
    let sigma, tycon' =
      match tycon with
      | None -> sigma, tycon
      | Some ty ->
        let sigma, ty' = Coercion.inh_coerce_to_prod ?loc ~program_mode !!env sigma ty in
        sigma, Some ty'
    in
    let sigma, (name',dom,rng) = split_tycon ?loc !!env sigma tycon' in
    let dom_valcon = valcon_of_tycon dom in
    let sigma, j = pretype_type dom_valcon env sigma c1 in
    let name = {binder_name=name; binder_relevance=Sorts.relevance_of_sort j.utj_type} in
    let var = LocalAssum (name, j.utj_val) in
    let hypnaming = if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames in
    let var',env' = push_rel ~hypnaming sigma var env in
    let sigma, j' = pretype rng env' sigma c2 in
    let name = get_name var' in
    let resj = judge_of_abstraction !!env (orelse_name name name'.binder_name) j j' in
    discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma resj tycon

  | GProd(name,bk,c1,c2) ->
    let sigma, j = pretype_type empty_valcon env sigma c1 in
    let hypnaming = if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames in
    let sigma, name, j' = match name with
      | Anonymous ->
        let sigma, j = pretype_type empty_valcon env sigma c2 in
        sigma, name, { j with utj_val = lift 1 j.utj_val }
      | Name _ ->
        let r = Sorts.relevance_of_sort j.utj_type in
        let var = LocalAssum (make_annot name r, j.utj_val) in
        let var, env' = push_rel ~hypnaming sigma var env in
        let sigma, c2_j = pretype_type empty_valcon env' sigma c2 in
        sigma, get_name var, c2_j
    in
    let resj =
      try
        judge_of_product !!env name j j'
      with TypeError _ as e ->
        let (e, info) = CErrors.push e in
        let info = Option.cata (Loc.add_loc info) info loc in
        iraise (e, info) in
      discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma resj tycon

  | GLetIn(name,c1,t,c2) ->
    let sigma, tycon1 =
      match t with
      | Some t ->
        let sigma, t_j = pretype_type empty_valcon env sigma t in
        sigma, mk_tycon t_j.utj_val
      | None ->
        sigma, empty_tycon in
    let sigma, j = pretype tycon1 env sigma c1 in
    let sigma, t = Evarsolve.refresh_universes
      ~onlyalg:true ~status:Evd.univ_flexible (Some false) !!env sigma j.uj_type in
    let r = Retyping.relevance_of_term !!env sigma j.uj_val in
    let var = LocalDef (make_annot name r, j.uj_val, t) in
    let tycon = lift_tycon 1 tycon in
    let hypnaming = if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames in
    let var, env = push_rel ~hypnaming sigma var env in
    let sigma, j' = pretype tycon env sigma c2 in
    let name = get_name var in
    sigma, { uj_val = mkLetIn (make_annot name r, j.uj_val, t, j'.uj_val) ;
             uj_type = subst1 j.uj_val j'.uj_type }

  | GLetTuple (nal,(na,po),c,d) ->
    let sigma, cj = pretype empty_tycon env sigma c in
    let (IndType (indf,realargs)) =
      try find_rectype !!env sigma cj.uj_type
      with Not_found ->
        let cloc = loc_of_glob_constr c in
          error_case_not_inductive ?loc:cloc !!env sigma cj
    in
    let ind = fst (fst (dest_ind_family indf)) in
    let cstrs = get_constructors !!env indf in
    if not (Int.equal (Array.length cstrs) 1) then
      user_err ?loc  (str "Destructing let is only for inductive types" ++
        str " with one constructor.");
    let cs = cstrs.(0) in
    if not (Int.equal (List.length nal) cs.cs_nargs) then
      user_err ?loc:loc (str "Destructing let on this type expects " ++
        int cs.cs_nargs ++ str " variables.");
    let fsign, record =
      let set_name na d = set_name na (map_rel_decl EConstr.of_constr d) in
      match Environ.get_projections !!env ind with
      | None ->
         List.map2 set_name (List.rev nal) cs.cs_args, false
      | Some ps ->
        let rec aux n k names l =
          match names, l with
          | na :: names, (LocalAssum (na', t) :: l) ->
            let t = EConstr.of_constr t in
            let proj = Projection.make ps.(cs.cs_nargs - k) true in
            LocalDef ({na' with binder_name = na},
                      lift (cs.cs_nargs - n) (mkProj (proj, cj.uj_val)), t)
            :: aux (n+1) (k + 1) names l
          | na :: names, (decl :: l) ->
            set_name na decl :: aux (n+1) k names l
          | [], [] -> []
          | _ -> assert false
        in aux 1 1 (List.rev nal) cs.cs_args, true in
    let fsign = Context.Rel.map (whd_betaiota sigma) fsign in
    let hypnaming = if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames in
    let fsign,env_f = push_rel_context ~hypnaming sigma fsign env in
    let obj ind rci p v f =
      if not record then
        let f = it_mkLambda_or_LetIn f fsign in
        let ci = make_case_info !!env (fst ind) rci LetStyle in
          mkCase (ci, p, cj.uj_val,[|f|])
      else it_mkLambda_or_LetIn f fsign
    in
    (* Make dependencies from arity signature impossible *)
    let arsgn, indr =
      let arsgn,s = get_arity !!env indf in
      List.map (set_name Anonymous) arsgn, Sorts.relevance_of_sort_family s
    in
      let indt = build_dependent_inductive !!env indf in
      let psign = LocalAssum (make_annot na indr, indt) :: arsgn in (* For locating names in [po] *)
      let psign = List.map (fun d -> map_rel_decl EConstr.of_constr d) psign in
      let predenv = Cases.make_return_predicate_ltac_lvar env sigma na c cj.uj_val in
      let nar = List.length arsgn in
      let psign',env_p = push_rel_context ~hypnaming ~force_names:true sigma psign predenv in
          (match po with
          | Some p ->
            let sigma, pj = pretype_type empty_valcon env_p sigma p in
            let ccl = nf_evar sigma pj.utj_val in
            let p = it_mkLambda_or_LetIn ccl psign' in
            let inst =
              (Array.map_to_list EConstr.of_constr cs.cs_concl_realargs)
              @[EConstr.of_constr (build_dependent_constructor cs)] in
            let lp = lift cs.cs_nargs p in
            let fty = hnf_lam_applist !!env sigma lp inst in
            let sigma, fj = pretype (mk_tycon fty) env_f sigma d in
            let v =
              let ind,_ = dest_ind_family indf in
                let rci = Typing.check_allowed_sort !!env sigma ind cj.uj_val p in
                obj ind rci p cj.uj_val fj.uj_val
            in
            sigma, { uj_val = v; uj_type = (substl (realargs@[cj.uj_val]) ccl) }

          | None ->
            let tycon = lift_tycon cs.cs_nargs tycon in
            let sigma, fj = pretype tycon env_f sigma d in
            let ccl = nf_evar sigma fj.uj_type in
            let ccl =
              if noccur_between sigma 1 cs.cs_nargs ccl then
                lift (- cs.cs_nargs) ccl
              else
                error_cant_find_case_type ?loc !!env sigma
                  cj.uj_val in
                 (* let ccl = refresh_universes ccl in *)
            let p = it_mkLambda_or_LetIn (lift (nar+1) ccl) psign' in
            let v =
              let ind,_ = dest_ind_family indf in
                let rci = Typing.check_allowed_sort !!env sigma ind cj.uj_val p in
                obj ind rci p cj.uj_val fj.uj_val
            in sigma, { uj_val = v; uj_type = ccl })

  | GIf (c,(na,po),b1,b2) ->
    let sigma, cj = pretype empty_tycon env sigma c in
    let (IndType (indf,realargs)) =
      try find_rectype !!env sigma cj.uj_type
      with Not_found ->
        let cloc = loc_of_glob_constr c in
          error_case_not_inductive ?loc:cloc !!env sigma cj in
    let cstrs = get_constructors !!env indf in
      if not (Int.equal (Array.length cstrs) 2) then
        user_err ?loc
                      (str "If is only for inductive types with two constructors.");

      let arsgn, indr =
        let arsgn,s = get_arity !!env indf in
        (* Make dependencies from arity signature impossible *)
        List.map (set_name Anonymous) arsgn, Sorts.relevance_of_sort_family s
      in
      let nar = List.length arsgn in
      let indt = build_dependent_inductive !!env indf in
      let psign = LocalAssum (make_annot na indr, indt) :: arsgn in (* For locating names in [po] *)
      let psign = List.map (fun d -> map_rel_decl EConstr.of_constr d) psign in
      let predenv = Cases.make_return_predicate_ltac_lvar env sigma na c cj.uj_val in
    let hypnaming = if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames in
      let psign,env_p = push_rel_context ~hypnaming sigma psign predenv in
      let sigma, pred, p = match po with
        | Some p ->
          let sigma, pj = pretype_type empty_valcon env_p sigma p in
          let ccl = nf_evar sigma pj.utj_val in
          let pred = it_mkLambda_or_LetIn ccl psign in
          let typ = lift (- nar) (beta_applist sigma (pred,[cj.uj_val])) in
          sigma, pred, typ
        | None ->
          let sigma, p = match tycon with
            | Some ty -> sigma, ty
            | None -> new_type_evar env sigma loc
          in
          sigma, it_mkLambda_or_LetIn (lift (nar+1) p) psign, p in
      let pred = nf_evar sigma pred in
      let p = nf_evar sigma p in
      let f sigma cs b =
        let n = Context.Rel.length cs.cs_args in
        let pi = lift n pred in (* liftn n 2 pred ? *)
        let pi = beta_applist sigma (pi, [EConstr.of_constr (build_dependent_constructor cs)]) in
        let cs_args = List.map (fun d -> map_rel_decl EConstr.of_constr d) cs.cs_args in
        let cs_args = Context.Rel.map (whd_betaiota sigma) cs_args in
        let csgn =
          List.map (set_name Anonymous) cs_args
        in
        let _,env_c = push_rel_context ~hypnaming sigma csgn env in
        let sigma, bj = pretype (mk_tycon pi) env_c sigma b in
        sigma, it_mkLambda_or_LetIn bj.uj_val cs_args in
      let sigma, b1 = f sigma cstrs.(0) b1 in
      let sigma, b2 = f sigma cstrs.(1) b2 in
      let v =
        let ind,_ = dest_ind_family indf in
        let pred = nf_evar sigma pred in
        let rci = Typing.check_allowed_sort !!env sigma ind cj.uj_val pred in
        let ci = make_case_info !!env (fst ind) rci IfStyle in
          mkCase (ci, pred, cj.uj_val, [|b1;b2|])
      in
      let cj = { uj_val = v; uj_type = p } in
      discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma cj tycon

  | GCases (sty,po,tml,eqns) ->
    Cases.compile_cases ?loc ~program_mode sty (pretype, sigma) tycon env (po,tml,eqns)

  | GCast (c,k) ->
    let sigma, cj =
      match k with
      | CastCoerce ->
        let sigma, cj = pretype empty_tycon env sigma c in
        Coercion.inh_coerce_to_base ?loc ~program_mode !!env sigma cj
      | CastConv t | CastVM t | CastNative t ->
        let k = (match k with CastVM _ -> VMcast | CastNative _ -> NATIVEcast | _ -> DEFAULTcast) in
        let sigma, tj = pretype_type empty_valcon env sigma t in
        let sigma, tval = Evarsolve.refresh_universes
            ~onlyalg:true ~status:Evd.univ_flexible (Some false) !!env sigma tj.utj_val in
        let tval = nf_evar sigma tval in
        let (sigma, cj), tval = match k with
          | VMcast ->
            let sigma, cj = pretype empty_tycon env sigma c in
            let cty = nf_evar sigma cj.uj_type and tval = nf_evar sigma tval in
              if not (occur_existential sigma cty || occur_existential sigma tval) then
                match Reductionops.vm_infer_conv !!env sigma cty tval with
                | Some sigma -> (sigma, cj), tval
                | None ->
                  error_actual_type ?loc !!env sigma cj tval
                      (ConversionFailed (!!env,cty,tval))
              else user_err ?loc  (str "Cannot check cast with vm: " ++
                str "unresolved arguments remain.")
          | NATIVEcast ->
            let sigma, cj = pretype empty_tycon env sigma c in
            let cty = nf_evar sigma cj.uj_type and tval = nf_evar sigma tval in
            begin
              match Nativenorm.native_infer_conv !!env sigma cty tval with
              | Some sigma -> (sigma, cj), tval
              | None ->
                error_actual_type ?loc !!env sigma cj tval
                  (ConversionFailed (!!env,cty,tval))
            end
          | _ ->
            pretype (mk_tycon tval) env sigma c, tval
        in
        let v = mkCast (cj.uj_val, k, tval) in
        sigma, { uj_val = v; uj_type = tval }
    in discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma cj tycon

      | GInt i ->
        let resj =
          try Typing.judge_of_int !!env i
          with Invalid_argument _ ->
            user_err ?loc ~hdr:"pretype" (str "Type of int63 should be registered first.")
        in
        discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma resj tycon
  | GFloat f ->
      let resj =
        try Typing.judge_of_float !!env f
        with Invalid_argument _ ->
          user_err ?loc ~hdr:"pretype" (str "Type of float should be registered first.")
        in
        discard_trace @@ inh_conv_coerce_to_tycon ?loc env sigma resj tycon

and pretype_instance ~program_mode ~poly resolve_tc env sigma loc hyps evk update =
  let f decl (subst,update,sigma) =
    let id = NamedDecl.get_id decl in
    let b = Option.map (replace_vars subst) (NamedDecl.get_value decl) in
    let t = replace_vars subst (NamedDecl.get_type decl) in
    let check_body sigma id c =
      match b, c with
      | Some b, Some c ->
         if not (is_conv !!env sigma b c) then
           user_err ?loc  (str "Cannot interpret " ++
             pr_existential_key sigma evk ++
             strbrk " in current context: binding for " ++ Id.print id ++
             strbrk " is not convertible to its expected definition (cannot unify " ++
             quote (Termops.Internal.print_constr_env !!env sigma b) ++
             strbrk " and " ++
             quote (Termops.Internal.print_constr_env !!env sigma c) ++
             str ").")
      | Some b, None ->
           user_err ?loc  (str "Cannot interpret " ++
             pr_existential_key sigma evk ++
             strbrk " in current context: " ++ Id.print id ++
             strbrk " should be bound to a local definition.")
      | None, _ -> () in
    let check_type sigma id t' =
      if not (is_conv !!env sigma t t') then
        user_err ?loc  (str "Cannot interpret " ++
          pr_existential_key sigma evk ++
          strbrk " in current context: binding for " ++ Id.print id ++
          strbrk " is not well-typed.") in
    let sigma, c, update =
      try
        let c = List.assoc id update in
        let sigma, c = pretype ~program_mode ~poly resolve_tc (mk_tycon t) env sigma c in
        check_body sigma id (Some c.uj_val);
        sigma, c.uj_val, List.remove_assoc id update
      with Not_found ->
      try
        let (n,b',t') = lookup_rel_id id (rel_context !!env) in
        check_type sigma id (lift n t');
        check_body sigma id (Option.map (lift n) b');
        sigma, mkRel n, update
      with Not_found ->
      try
        let decl = lookup_named id !!env in
        check_type sigma id (NamedDecl.get_type decl);
        check_body sigma id (NamedDecl.get_value decl);
        sigma, mkVar id, update
      with Not_found ->
        user_err ?loc  (str "Cannot interpret " ++
          pr_existential_key sigma evk ++
          str " in current context: no binding for " ++ Id.print id ++ str ".") in
    ((id,c)::subst, update, sigma) in
  let subst,inst,sigma = List.fold_right f hyps ([],update,sigma) in
  check_instance loc subst inst;
  sigma, Array.map_of_list snd subst

(* [pretype_type valcon env sigma c] coerces [c] into a type *)
and pretype_type ~program_mode ~poly resolve_tc valcon (env : GlobEnv.t) sigma c = match DAst.get c with
  | GHole (knd, naming, None) ->
      let loc = loc_of_glob_constr c in
      (match valcon with
       | Some v ->
           let sigma, s =
             let t = Retyping.get_type_of !!env sigma v in
               match EConstr.kind sigma (whd_all !!env sigma t) with
               | Sort s ->
                 sigma, ESorts.kind sigma s
               | Evar ev when is_Type sigma (existential_type sigma ev) ->
                 define_evar_as_sort !!env sigma ev
               | _ -> anomaly (Pp.str "Found a type constraint which is not a type.")
           in
           (* Correction of bug #5315 : we need to define an evar for *all* holes *)
           let sigma, evkt = new_evar env sigma ~src:(loc, knd) ~naming (mkSort s) in
           let ev,_ = destEvar sigma evkt in
           let sigma = Evd.define ev (nf_evar sigma v) sigma in
           (* End of correction of bug #5315 *)
           sigma, { utj_val = v;
                    utj_type = s }
       | None ->
         let sigma, s = new_sort_variable univ_flexible_alg sigma in
         let sigma, utj_val = new_evar env sigma ~src:(loc, knd) ~naming (mkSort s) in
         let sigma = if program_mode then mark_obligation_evar sigma knd utj_val else sigma in
         sigma, { utj_val; utj_type = s})
  | _ ->
      let sigma, j = pretype ~program_mode ~poly resolve_tc empty_tycon env sigma c in
      let loc = loc_of_glob_constr c in
      let sigma, tj = Coercion.inh_coerce_to_sort ?loc !!env sigma j in
        match valcon with
        | None -> sigma, tj
        | Some v ->
          begin match Evarconv.unify_leq_delay !!env sigma v tj.utj_val with
            | sigma -> sigma, tj
            | exception Evarconv.UnableToUnify _ ->
              error_unexpected_type
                ?loc:(loc_of_glob_constr c) !!env sigma tj.utj_val v
          end

let ise_pretype_gen flags env sigma lvar kind c =
  let program_mode = flags.program_mode in
  let poly = flags.polymorphic in
  let hypnaming =
    if program_mode then ProgramNaming else KeepUserNameAndRenameExistingButSectionNames
  in
  let env = GlobEnv.make ~hypnaming env sigma lvar in
  let sigma', c', c'_ty = match kind with
    | WithoutTypeConstraint ->
      let sigma, j = pretype ~program_mode ~poly flags.use_typeclasses empty_tycon env sigma c in
      sigma, j.uj_val, j.uj_type
    | OfType exptyp ->
      let sigma, j = pretype ~program_mode ~poly flags.use_typeclasses (mk_tycon exptyp) env sigma c in
      sigma, j.uj_val, j.uj_type
    | IsType ->
      let sigma, tj = pretype_type ~program_mode ~poly flags.use_typeclasses empty_valcon env sigma c in
      sigma, tj.utj_val, mkSort tj.utj_type
  in
  process_inference_flags flags !!env sigma (sigma',c',c'_ty)

let default_inference_flags fail = {
  use_typeclasses = true;
  solve_unification_constraints = true;
  fail_evar = fail;
  expand_evars = true;
  program_mode = false;
  polymorphic = false;
}

let no_classes_no_fail_inference_flags = {
  use_typeclasses = false;
  solve_unification_constraints = true;
  fail_evar = false;
  expand_evars = true;
  program_mode = false;
  polymorphic = false;
}

let all_and_fail_flags = default_inference_flags true
let all_no_fail_flags = default_inference_flags false

let ise_pretype_gen_ctx flags env sigma lvar kind c =
  let sigma, c, _ = ise_pretype_gen flags env sigma lvar kind c in
  c, Evd.evar_universe_context sigma

(** Entry points of the high-level type synthesis algorithm *)

let understand
    ?(flags=all_and_fail_flags)
    ?(expected_type=WithoutTypeConstraint)
    env sigma c =
  ise_pretype_gen_ctx flags env sigma empty_lvar expected_type c

let understand_tcc_ty ?(flags=all_no_fail_flags) env sigma ?(expected_type=WithoutTypeConstraint) c =
  ise_pretype_gen flags env sigma empty_lvar expected_type c

let understand_tcc ?flags env sigma ?expected_type c =
  let sigma, c, _ = understand_tcc_ty ?flags env sigma ?expected_type c in
  sigma, c

let understand_ltac flags env sigma lvar kind c =
  let (sigma, c, _) = ise_pretype_gen flags env sigma lvar kind c in
  (sigma, c)

let path_convertible env sigma p q =
  let open Classops in
  let mkGRef ref          = DAst.make @@ Glob_term.GRef(ref,None) in
  let mkGVar id           = DAst.make @@ Glob_term.GVar(id) in
  let mkGApp(rt,rtl)      = DAst.make @@ Glob_term.GApp(rt,rtl) in
  let mkGLambda(n,t,b)    = DAst.make @@ Glob_term.GLambda(n,Explicit,t,b) in
  let mkGHole ()          = DAst.make @@ Glob_term.GHole(Evar_kinds.BinderType Anonymous,Namegen.IntroAnonymous,None) in
  let path_to_gterm p =
    match p with
    | ic :: p' ->
      let names =
        List.map (fun n -> Id.of_string ("x" ^ string_of_int n))
          (List.interval 0 ic.coe_param)
      in
      List.fold_right
        (fun id t -> mkGLambda (Name id, mkGHole (), t)) names @@
        List.fold_left
          (fun t ic ->
             mkGApp (mkGRef ic.coe_value,
                     List.make ic.coe_param (mkGHole ()) @ [t]))
          (mkGApp (mkGRef ic.coe_value, List.map (fun i -> mkGVar i) names))
          p'
    | [] -> anomaly (str "A coercion path shouldn't be empty.")
  in
  try
    let sigma,tp = understand_tcc env sigma (path_to_gterm p) in
    let sigma,tq = understand_tcc env sigma (path_to_gterm q) in
    if Evd.has_undefined sigma then
      false
    else
      let _ = Evarconv.unify_delay env sigma tp tq in true
  with Evarconv.UnableToUnify _ | PretypeError _ -> false

let _ = Classops.install_path_comparator path_convertible
