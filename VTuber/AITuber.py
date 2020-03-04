import unreal_engine as ue
import tensorflow as tf
import os
from .SentimentAnalysis import SentimentAnalysis
from .Conversation import Conversation
from .Speaker import Speaker
from .YouTubeLive import YouTubeLive

graph = tf.get_default_graph()

SA_TOKENIZER = os.path.join(ue.get_content_dir(), 'tokenizer.pkl')
SA_MODEL = os.path.join(ue.get_content_dir(), 'model.h5')
CONVERSATION_TOKENIZER = os.path.join(
    ue.get_content_dir(), 'conv_tokenizer.pkl')
CONVERSATION_MODEL = os.path.join(ue.get_content_dir(), 'conv_model.h5')
DEVELOPER_KEY = 'REPLACE_ME'
CHANNEL_ID = 'REPLACE_ME'


class AITuber:
    def __init__(self):
        global graph
        with graph.as_default():
            self.sentiment_analysis = SentimentAnalysis(
                SA_TOKENIZER, SA_MODEL)
            self.conversation = Conversation(
                CONVERSATION_TOKENIZER, CONVERSATION_MODEL)
        self.speaker = Speaker()
        self.you_tube_live = YouTubeLive(DEVELOPER_KEY, CHANNEL_ID)

    def begin_play(self):
        pass

    def fetch_last_message(self):
        return self.you_tube_live.fetch_last_live_message()

    def predict_sentiment_label(self, text):
        sentiment_analysis_result = self.sentiment_analysis.predict(text)
        ue.log(sentiment_analysis_result)
        return sentiment_analysis_result['label']

    def reply(self, text):
        conversation_result = self.conversation.reply(text)
        ue.log(conversation_result)
        return conversation_result

    def speak(self, text):
        ue.log(text)
        self.speaker.speak(text)
        return

    def tick(self, delta_time):
        pass
