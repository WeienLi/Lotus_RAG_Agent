# #from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
# history = ChatMessageHistory(len = 1)

# history.add_user_message(HumanMessage(content="hi", id = 123))

# history.add_ai_message("whats up?")

# history.add_user_message(HumanMessage(content="Dog", id = 243))

# history.add_ai_message("Shut up?")
# print(history)
# #.dict()['messages'][0] .messages[0].id

from collections import OrderedDict

class FixedSizeDict(OrderedDict):
    def __init__(self, max_length, *args, **kwargs):
        self.max_length = max_length
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        if len(self) >= self.max_length:
            self.popitem(last=False)  # Remove the oldest item (first inserted)
        super().__setitem__(key, value)

# Usage example
fixed_size_dict = FixedSizeDict(max_length=3)
fixed_size_dict['a'] = 1
fixed_size_dict['b'] = 2
fixed_size_dict['c'] = 3
print(fixed_size_dict)  # Output: OrderedDict([('a', 1), ('b', 2), ('c', 3)])

# Adding another item, which should remove the oldest item ('a')
fixed_size_dict['d'] = 4
print(fixed_size_dict)  # Output: OrderedDict([('b', 2), ('c', 3), ('d', 4)])
