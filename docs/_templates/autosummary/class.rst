{{ ("``" ~ (objname | escape) ~ "``") | underline('=')}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   
   {% block methods %}
   
   {% set methods_list = [] %}
   {% for item in all_methods %}
      {%- if not item.startswith('__') and not item in inherited_members %}
         {% set methods_list = methods_list.append( item ) %}
      {%- endif -%}
   {%- endfor %}
   
   {% if methods_list %}
   .. rubric:: {{ _('Methods') }}
   
   {% if methods_list|length > 1 %}
   .. autosummary::
      :signatures: none
      {% for item in methods_list %}
      {{ item }}
      {%- endfor %}
   {% endif %}

   {% for item in methods_list %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set attributes_list = [] %}
   {% for item in all_attributes %}
      {%- if not item.startswith('_') and not item in inherited_members %}
      {% set attributes_list = attributes_list.append( item ) %}
      {%- endif -%}
   {%- endfor %}

   {% if attributes_list %}
   .. rubric:: {{ _('Attributes') }}

   {% if attributes_list|length > 1 %}
   .. autosummary::
      {% for item in attributes_list %}
      {{ item }}
      {%- endfor %}
   {% endif %}
   
   {% for item in attributes_list %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
