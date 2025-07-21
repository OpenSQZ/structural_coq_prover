from .Def_class import VernacBase, ConstrBase

@VernacBase.register_subclass("Definition")
class VernacDefinition(VernacBase):
    pass

@ConstrBase.register_subclass("Definition")
class ConstrDefinition(ConstrBase):
    pass

