"""Read-only scene diagnostics, invoked by Blender after loading a .blend file."""
import bpy
for o in bpy.context.scene.objects:
    print('OBJECT', o.name, o.type, 'hide_render', o.hide_render,
          'parent', o.parent.name if o.parent else None)
