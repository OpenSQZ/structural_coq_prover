from .Def_class import ConstrBase

## Primitive do not have Vernac
## TODO

@ConstrBase.register_subclass("Primitive")
class ConstrPrimitive(ConstrBase):
    pass