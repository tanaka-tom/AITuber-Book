import unreal_engine as ue
import time
from googleapiclient.discovery import build


DEVELOPER_KEY = 'REPLACE_ME'
CHANNEL_ID = 'REPLACE_ME'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


class Ue4YouTubeLive:
    def __init__(self):
        self.youtube_live = YouTubeLive(DEVELOPER_KEY, CHANNEL_ID)

    def begin_play(self):
        self.youtube_live.fetch_last_live_message()


class YouTubeLive:
    def __init__(self, developer_key, channel_id):
        self.youtube = build(YOUTUBE_API_SERVICE_NAME,
                             YOUTUBE_API_VERSION,
                             developerKey=developer_key)

        search_response = self.youtube.search().list(
            channelId=channel_id,
            type='video',
            eventType='live',
            part='id',
        ).execute()

        live = search_response.get('items', [])[0]

        live_video_id = live['id']['videoId']

        video_response = self.youtube.videos().list(
            part="liveStreamingDetails,snippet",
            id=live_video_id).execute()

        video = video_response.get('items', [])[0]
        live_chat_id = video['liveStreamingDetails']['activeLiveChatId']

        self.active_live_chat_id = live_chat_id

    def fetch_last_live_message(self):
        start_time = time.time()
        message_response = self.youtube.liveChatMessages().list(
            liveChatId=self.active_live_chat_id,
            part='snippet').execute()

        message = message_response.get(
            'items', [])[0]['snippet']['displayMessage']

        ue.log(message)
        ue.log("fetch_last_live_message: {}".format(
            time.time() - start_time))
        return message
