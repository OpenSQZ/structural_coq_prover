from .Def_class import VernacBase, ConstrBase

@VernacBase.register_subclass("Assumption")
class VernacAssumption(VernacBase):
    pass

@ConstrBase.register_subclass("Assumption")
class ConstrAssumption(ConstrBase):
    pass
