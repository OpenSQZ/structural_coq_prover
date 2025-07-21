from .Def_class import VernacBase, ConstrBase

@VernacBase.register_subclass("Proof")
class VernacProof(VernacBase):
    pass

@ConstrBase.register_subclass("Proof")
class ConstrProof(ConstrBase):
    pass
