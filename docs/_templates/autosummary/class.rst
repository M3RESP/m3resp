{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:

   {% block methods %}
   {% set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list %}
   {% if own_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in own_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set own_attributes = attributes | reject("in", inherited_members) | list %}
   {% if own_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in own_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. minigallery:: {{ fullname }}
   :add-heading: Tutorials that use ``{{ objname }}``
   :heading-level: -
