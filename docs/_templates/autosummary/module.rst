{{ objname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      {% for item in attributes %}
      {{ item }}
      {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :signatures: none
      {% for item in functions %}
      {{ item }}
      {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :signatures: none
      :toctree:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :signatures: none
      {% for item in exceptions %}
      {{ item }}
      {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree: 
   :recursive:
   {% for item in modules %}
   {{ item }}
   {%- endfor %}
{% endif %}
{%- endblock %}
