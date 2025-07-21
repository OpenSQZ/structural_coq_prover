from .Def_class import VernacBase, ConstrBase

# do not have vernac representation

@ConstrBase.register_subclass("Arguments")
class ConstrArguments(ConstrBase):
    pass