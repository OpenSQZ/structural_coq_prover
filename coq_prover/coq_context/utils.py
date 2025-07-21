from .prompt import INTERNAL_ORIGIN_MIXED_FORMAT

## in def_table, some can not find the origin_context
## like arguments/cofixpoint/constructors have no idea
## Primitive do not need !!!!!!!!!!!!!!!
def format_def(def_, use_origin='mixed'):
    if use_origin == 'empty':
        return ''
    
    if def_['kind'] == 'Primitive':
        return ''
    if def_['kind'] == 'Ltac':
        def_text=def_['origin_context']['content']
        if len(def_text) > 4000:
            def_text = def_text[:2000] + '...' + def_text[-2000:]
    else:
        def_origin = def_['origin_context']['content'] if def_['origin_context'] else ''
        def_internal_body = ''
        if def_['internal_context']['body'] and def_['internal_context']['body']['processed']['origin'].strip():
            def_internal_body = 'body: ' + def_['internal_context']['body']['processed']['origin']
        def_internal = def_['internal_context']['content']['processed']['origin'] + def_internal_body
        if use_origin == 'origin':
            def_text = def_origin
            if len(def_text) > 4000:
                def_text = def_text[:2000] + '...' + def_text[-2000:]
        elif use_origin == 'internal':
            def_text = def_internal
            if len(def_text) > 4000:
                def_text = def_text[:2000] + '...' + def_text[-2000:]
        elif use_origin == 'mixed':
            if len(def_internal) > 4000:
                def_internal = def_internal[:2000] + '...' + def_internal[-2000:]
            if len(def_origin) > 4000:
                def_origin = def_origin[:2000] + '...' + def_origin[-2000:]
            def_text = INTERNAL_ORIGIN_MIXED_FORMAT.format(internal=def_internal, origin=def_origin)
        else:
            raise ValueError(f"Invalid use_origin: {use_origin}")
    return def_text