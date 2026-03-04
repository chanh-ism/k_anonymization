from ipywidgets import TagsInput
from traitlets import observe


class AutoSortedTagsInput(TagsInput):

    @observe("value")
    def _sort_value(self, change):
        if not change.old:
            return
        if len(change.new) < len(change.old):
            return
        self.value = [i for i in self.allowed_tags if i in change.new]

    @observe("allowed_tags")
    def _remove_unallowed_tags(self, change):
        if not change.old:
            return
        if len(change.new) > len(change.old):
            return
        self.value = [i for i in change.new if i in self.value]
