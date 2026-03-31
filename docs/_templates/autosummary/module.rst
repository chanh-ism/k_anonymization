{{ (":code:`" ~ (objname | escape) ~ "`") | underline('=')}}

.. automodule:: {{ fullname }}

   {% block types %}
   {%- if objname == "local_recoding" %}
   .. rubric:: {{ _('Type Alias') }}
   
   .. include:: /_manual_api_reference/GroupAnonymization.rst
         
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

   {%- block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   {% if attributes|length > 1 %}
   .. autosummary::
      {% for item in attributes %}
      {{ item }}
      {%- endfor %}
   {% endif %}

   {% for item in attributes %}
   .. autodata:: {{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   {% if functions|length > 1 %}
   .. autosummary::
      :signatures: none
      {% for item in functions %}
      {{ item }}
      {%- endfor %}
   {% endif %}

   {% for item in functions %}
   .. autofunction:: {{ fullname }}.{{ item }}
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
