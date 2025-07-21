from .Def_class import VernacBase, ConstrBase

@VernacBase.register_subclass("Instance")
class VernacInstance(VernacBase):
    pass

@ConstrBase.register_subclass("Instance")
class ConstrInstance(ConstrBase):
    pass