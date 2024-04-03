import copy
import os
import re
import subprocess
import tempfile
import time
import traceback
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

import iso639
import psutil
import pytz
import srt
from apscheduler.schedulers.background import BackgroundScheduler
from lxml import etree

from app.core.config import settings
from app.log import logger
from app.plugins import _PluginBase
from app.utils.system import SystemUtils
from plugins.autosub.ffmpeg import Ffmpeg
from plugins.autosub.translate.openai import OpenAi


class AutoSub(_PluginBase):
    # 插件名称
    plugin_name = "AI字幕自动生成"
    # 插件描述
    plugin_desc = "使用whisper自动生成视频文件字幕。"
    # 插件图标
    plugin_icon = "autosubtitles.jpeg"
    # 主题色
    plugin_color = "#2C4F7E"
    # 插件版本
    plugin_version = "0.6"
    # 插件作者
    plugin_author = "olly"
    # 作者主页
    author_url = "https://github.com/lightolly"
    # 插件配置项ID前缀
    plugin_config_prefix = "autosub"
    # 加载顺序
    plugin_order = 14
    # 可使用的用户级别
    auth_level = 2

    # 私有属性
    _running = False
    # 语句结束符
    _end_token = ['.', '!', '?', '。', '！', '？', '。"', '！"', '？"', '."', '!"', '?"']
    _noisy_token = [('(', ')'), ('[', ']'), ('{', '}'), ('【', '】'), ('♪', '♪'), ('♫', '♫'), ('♪♪', '♪♪')]

    def __init__(self):
        super().__init__()
        # ChatGPT
        self.openai = None
        self._chatgpt = None
        self._openai_key = None
        self._openai_url = None
        self._openai_proxy = None
        self._openai_model = None
        self._scheduler = None
        self.process_count = None
        self.fail_count = None
        self.success_count = None
        self.skip_count = None
        self.faster_whisper_model_path = None
        self.faster_whisper_model = None
        self.asr_engine = None
        self.send_notify = None
        self.additional_args = None
        self.translate_only = None
        self.translate_zh = None
        self.whisper_model = None
        self.whisper_main = None
        self.file_size = None

    def init_plugin(self, config=None):
        self.additional_args = '-t 4 -p 1'
        self.translate_zh = False
        self.translate_only = False
        self.whisper_model = None
        self.whisper_main = None
        self.file_size = None
        self.process_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.success_count = 0
        self.send_notify = False
        self.asr_engine = 'whisper.cpp'
        self.faster_whisper_model = 'base'
        self.faster_whisper_model_path = None

        # 如果没有配置信息， 则不处理
        if not config:
            return

        self.translate_zh = config.get('translate_zh', False)
        if self.translate_zh:
            chatgpt = self.get_config("ChatGPT")
            if not chatgpt:
                logger.error(f"翻译依赖于ChatGPT，请先维护ChatGPT插件")
                return
            self._chatgpt = chatgpt and chatgpt.get("enabled")
            self._openai_key = chatgpt and chatgpt.get("openai_key")
            self._openai_url = chatgpt and chatgpt.get("openai_url")
            self._openai_proxy = chatgpt and chatgpt.get("proxy")
            self._openai_model = chatgpt and chatgpt.get("model")
            if not self._openai_key:
                logger.error(f"翻译依赖于ChatGPT，请先维护openai_key")
                return
            self.openai = OpenAi(api_key=self._openai_key, api_url=self._openai_url,
                                 proxy=settings.PROXY if self._openai_proxy else None,
                                 model=self._openai_model)

        # config.get('path_list') 用 \n 分割为 list 并去除重复值和空值
        path_list = list(set(config.get('path_list').split('\n')))
        # file_size 转成数字
        self.file_size = config.get('file_size')
        self.whisper_main = config.get('whisper_main')
        self.whisper_model = config.get('whisper_model')
        self.translate_only = config.get('translate_only', False)
        self.additional_args = config.get('additional_args', '-t 4 -p 1')
        self.send_notify = config.get('send_notify', False)
        self.asr_engine = config.get('asr_engine', 'faster_whisper')
        self.faster_whisper_model = config.get('faster_whisper_model', 'base')
        self.faster_whisper_model_path = config.get('faster_whisper_model_path',
                                                    self.get_data_path() / "faster-whisper-models")

        run_now = config.get('run_now')
        if not run_now:
            return

        config['run_now'] = False
        self.update_config(config)

        # 如果没有配置信息， 则不处理
        if not path_list or not self.file_size:
            logger.warn(f"配置信息不完整，不进行处理")
            return

        # 校验文件大小是否为数字
        if not self.file_size.isdigit():
            logger.warn(f"文件大小不是数字，不进行处理")
            return

        # asr 配置检查
        if not self.translate_only and not self.__check_asr():
            return

        if self._running:
            logger.warn(f"上一次任务还未完成，不进行处理")
            return

        if run_now:
            self._scheduler = BackgroundScheduler(timezone=settings.TZ)
            logger.info("AI字幕自动生成任务，立即运行一次")
            self._scheduler.add_job(func=self._do_autosub, kwargs={'path_list': path_list}, trigger='date',
                                    run_date=datetime.now(tz=pytz.timezone(settings.TZ)) + timedelta(seconds=3),
                                    name="AI字幕自动生成")

            # 启动任务
            if self._scheduler.get_jobs():
                self._scheduler.print_jobs()
                self._scheduler.start()

    def _do_autosub(self, path_list: str):
        # 依次处理每个目录
        try:
            self._running = True
            self.success_count = self.skip_count = self.fail_count = self.process_count = 0
            for path in path_list:
                logger.info(f"开始处理目录：{path} ...")
                # 如果目录不存在， 则不处理
                if not os.path.exists(path):
                    logger.warn(f"目录不存在，不进行处理")
                    continue

                # 如果目录不是文件夹， 则不处理
                if not os.path.isdir(path):
                    logger.warn(f"目录不是文件夹，不进行处理")
                    continue

                # 如果目录不是绝对路径， 则不处理
                if not os.path.isabs(path):
                    logger.warn(f"目录不是绝对路径，不进行处理")
                    continue

                # 处理目录
                self.__process_folder_subtitle(path)
        except Exception as e:
            logger.error(f"处理异常: {e}")
        finally:
            logger.info(f"处理完成: "
                        f"成功{self.success_count} / 跳过{self.skip_count} / 失败{self.fail_count} / 共{self.process_count}")
            self._running = False

    def __check_asr(self):
        if self.asr_engine == 'whisper.cpp':
            if not self.whisper_main or not self.whisper_model:
                logger.warn(f"配置信息不完整，不进行处理")
                return
            if not os.path.exists(self.whisper_main):
                logger.warn(f"whisper.cpp主程序不存在，不进行处理")
                return False
            if not os.path.exists(self.whisper_model):
                logger.warn(f"whisper.cpp模型文件不存在，不进行处理")
                return False
            # 校验扩展参数是否包含异常字符
            if self.additional_args and re.search(r'[;|&]', self.additional_args):
                logger.warn(f"扩展参数包含异常字符，不进行处理")
                return False
        elif self.asr_engine == 'faster-whisper':
            if not self.faster_whisper_model_path or not self.faster_whisper_model:
                logger.warn(f"配置信息不完整，不进行处理")
                return
            if not os.path.exists(self.faster_whisper_model_path):
                logger.info(f"创建faster-whisper模型目录：{self.faster_whisper_model_path}")
                os.mkdir(self.faster_whisper_model_path)
            try:
                from faster_whisper import WhisperModel, download_model
            except ImportError:
                logger.warn(f"faster-whisper 未安装，不进行处理")
                return False
            return True
        else:
            logger.warn(f"未配置asr引擎，不进行处理")
            return False
        return True

    def __process_folder_subtitle(self, path):
        """
        处理目录字幕
        :param path:
        :return:
        """
        # 获取目录媒体文件列表
        for video_file in self.__get_library_files(path):
            if not video_file:
                continue
            # 如果文件大小小于指定大小， 则不处理
            if os.path.getsize(video_file) < int(self.file_size):
                continue

            self.process_count += 1
            start_time = time.time()
            file_path, file_ext = os.path.splitext(video_file)
            file_name = os.path.basename(video_file)

            try:
                logger.info(f"开始处理文件：{video_file} ...")
                # 判断目的字幕（和内嵌）是否已存在
                if self.__target_subtitle_exists(video_file):
                    logger.warn(f"字幕文件已经存在，不进行处理")
                    self.skip_count += 1
                    continue
                # 生成字幕
                if self.send_notify:
                    self.post_message(title="自动字幕生成",
                                      text=f" 媒体: {file_name}\n 开始处理文件 ... ")
                ret, lang = self.__generate_subtitle(video_file, file_path, self.translate_only)
                if not ret:
                    message = f" 媒体: {file_name}\n "
                    if self.translate_only:
                        message += "内嵌&外挂字幕不存在，不进行翻译"
                        self.skip_count += 1
                    else:
                        message += "生成字幕失败，跳过后续处理"
                        self.fail_count += 1

                    if self.send_notify:
                        self.post_message(title="自动字幕生成", text=message)
                    continue

                if self.translate_zh:
                    # 翻译字幕
                    logger.info(f"开始翻译字幕为中文 ...")
                    if self.send_notify:
                        self.post_message(title="自动字幕生成",
                                          text=f" 媒体: {file_name}\n 开始翻译字幕为中文 ... ")
                    self.__translate_zh_subtitle(lang, f"{file_path}.{lang}.srt", f"{file_path}.zh.srt")
                    logger.info(f"翻译字幕完成：{file_name}.zh.srt")

                end_time = time.time()
                message = f" 媒体: {file_name}\n 处理完成\n 字幕原始语言: {lang}\n "
                if self.translate_zh:
                    message += f"字幕翻译语言: zh\n "
                message += f"耗时：{round(end_time - start_time, 2)}秒"
                logger.info(f"自动字幕生成 处理完成：{message}")
                if self.send_notify:
                    self.post_message(title="自动字幕生成", text=message)
                self.success_count += 1
            except Exception as e:
                logger.error(f"自动字幕生成 处理异常：{e}")
                end_time = time.time()
                message = f" 媒体: {file_name}\n 处理失败\n 耗时：{round(end_time - start_time, 2)}秒"
                if self.send_notify:
                    self.post_message(title="自动字幕生成", text=message)
                # 打印调用栈
                traceback.print_exc()
                self.fail_count += 1

    def __do_speech_recognition(self, audio_lang, audio_file):
        """
        语音识别, 生成字幕
        :param audio_lang:
        :param audio_file:
        :return:
        """
        lang = audio_lang
        if self.asr_engine == 'whisper.cpp':
            command = [self.whisper_main] + self.additional_args.split()
            command += ['-l', lang, '-m', self.whisper_model, '-osrt', '-of', audio_file, audio_file]
            ret = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if ret.returncode == 0:
                if lang == 'auto':
                    # 从output中获取语言 "whisper_full_with_state: auto-detected language: en (p = 0.973642)"
                    output = ret.stdout.decode('utf-8') if ret.stdout else ""
                    lang = re.search(r"auto-detected language: (\w+)", output)
                    if lang and lang.group(1):
                        lang = lang.group(1)
                    else:
                        lang = "en"
                return True, lang
        elif self.asr_engine == 'faster-whisper':
            try:
                from faster_whisper import WhisperModel, download_model
                # 设置缓存目录, 防止缓存同目录出现 cross-device 错误
                cache_dir = os.path.join(self.faster_whisper_model_path, "cache")
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                os.environ["HF_HUB_CACHE"] = cache_dir
                model = WhisperModel(download_model(self.faster_whisper_model, cache_dir=cache_dir),
                                     device="cpu", compute_type="int8", cpu_threads=psutil.cpu_count(logical=False))
                segments, info = model.transcribe(audio_file,
                                                  language=lang if lang != 'auto' else None,
                                                  word_timestamps=True,
                                                  vad_filter=True,
                                                  temperature=0,
                                                  beam_size=5)
                if lang == 'auto':
                    lang = info.language

                subs = []
                if lang in ['en', 'eng']:
                    # 英文先生成单词级别字幕，再合并
                    idx = 0
                    for segment in segments:
                        for word in segment.words:
                            idx += 1
                            subs.append(srt.Subtitle(index=idx,
                                                     start=timedelta(seconds=word.start),
                                                     end=timedelta(seconds=word.end),
                                                     content=word.word))
                    subs = self.__merge_srt(subs)
                else:
                    for i, segment in enumerate(segments):
                        subs.append(srt.Subtitle(index=i,
                                                 start=timedelta(seconds=segment.start),
                                                 end=timedelta(seconds=segment.end),
                                                 content=segment.text))

                self.__save_srt(f"{audio_file}.srt", subs)
                return True, lang
            except ImportError:
                logger.warn(f"faster-whisper 未安装，不进行处理")
                return False, None
            except Exception as e:
                traceback.print_exc()
                logger.error(f"faster-whisper 处理异常：{e}")
                return False, None
        return False, None

    def __generate_subtitle(self, video_file, subtitle_file, only_extract=False):
        """
        生成字幕
        :param video_file: 视频文件
        :param subtitle_file: 字幕文件, 不包含后缀
        :return: 生成成功返回True，字幕语言，否则返回False, None
        """
        # 获取文件元数据
        video_meta = Ffmpeg().get_video_metadata(video_file)
        if not video_meta:
            logger.error(f"获取视频文件元数据失败，跳过后续处理")
            return False, None

        # 获取视频文件音轨和语言信息
        ret, audio_index, audio_lang = self.__get_video_prefer_audio(video_meta)
        if not ret:
            return False, None

        if not iso639.find(audio_lang) or not iso639.to_iso639_1(audio_lang):
            logger.info(f"未知语言音轨")
            audio_lang = 'auto'

        expert_subtitle_langs = ['en', 'eng'] if audio_lang == 'auto' else [audio_lang, iso639.to_iso639_1(audio_lang)]
        logger.info(f"使用 {expert_subtitle_langs} 匹配已有外挂字幕文件 ...")

        exist, lang = self.__external_subtitle_exists(video_file, expert_subtitle_langs)
        if exist:
            logger.info(f"外挂字幕文件已经存在，字幕语言 {lang}")
            return True, iso639.to_iso639_1(lang)

        logger.info(f"外挂字幕文件不存在，使用 {expert_subtitle_langs} 匹配内嵌字幕文件 ...")
        # 获取视频文件字幕信息
        ret, subtitle_index, \
            subtitle_lang, subtitle_count = self.__get_video_prefer_subtitle(video_meta, expert_subtitle_langs)
        if ret and (audio_lang == subtitle_lang or subtitle_count == 1):
            if audio_lang == subtitle_lang:
                # 如果音轨和字幕语言一致， 则直接提取字幕
                logger.info(f"内嵌音轨和字幕语言一致，直接提取字幕 ...")
            elif subtitle_count == 1:
                # 如果音轨和字幕语言不一致，但只有一个字幕， 则直接提取字幕
                logger.info(f"内嵌音轨和字幕语言不一致，但只有一个字幕，直接提取字幕 ...")

            audio_lang = iso639.to_iso639_1(subtitle_lang) \
                if (subtitle_lang and iso639.find(subtitle_lang) and iso639.to_iso639_1(subtitle_lang)) else 'und'
            Ffmpeg().extract_subtitle_from_video(video_file, f"{subtitle_file}.{audio_lang}.srt", subtitle_index)
            logger.info(f"提取字幕完成：{subtitle_file}.{audio_lang}.srt")
            return True, audio_lang

        if audio_lang != 'auto':
            audio_lang = iso639.to_iso639_1(audio_lang)

        if only_extract:
            logger.info(f"未开启语音识别，且无已有字幕文件，跳过后续处理")
            return False, None

        # 清理异常退出的临时文件
        tempdir = tempfile.gettempdir()
        for file in os.listdir(tempdir):
            if file.startswith('autosub-'):
                os.remove(os.path.join(tempdir, file))

        with tempfile.NamedTemporaryFile(prefix='autosub-', suffix='.wav', delete=True) as audio_file:
            # 提取音频
            logger.info(f"提取音频：{audio_file.name} ...")
            Ffmpeg().extract_wav_from_video(video_file, audio_file.name, audio_index)
            logger.info(f"提取音频完成：{audio_file.name}")

            # 生成字幕
            logger.info(f"开始生成字幕, 语言 {audio_lang} ...")
            ret, lang = self.__do_speech_recognition(audio_lang, audio_file.name)
            if ret:
                logger.info(f"生成字幕成功，原始语言：{lang}")
                # 复制字幕文件
                SystemUtils.copy(Path(f"{audio_file.name}.srt"), Path(f"{subtitle_file}.{lang}.srt"))
                logger.info(f"复制字幕文件：{subtitle_file}.{lang}.srt")
                # 删除临时文件
                os.remove(f"{audio_file.name}.srt")
                return ret, lang
            else:
                logger.error(f"生成字幕失败")
                return False, None

    @staticmethod
    def __get_library_files(in_path, exclude_path=None):
        """
        获取目录媒体文件列表
        """
        if not os.path.isdir(in_path):
            yield in_path
            return

        for root, dirs, files in os.walk(in_path):
            if exclude_path and any(os.path.abspath(root).startswith(os.path.abspath(path))
                                    for path in exclude_path.split(",")):
                continue

            for file in files:
                cur_path = os.path.join(root, file)
                # 检查后缀
                if os.path.splitext(file)[-1].lower() in settings.RMT_MEDIAEXT:
                    yield cur_path

    @staticmethod
    def __load_srt(file_path):
        """
        加载字幕文件
        :param file_path: 字幕文件路径
        :return:
        """
        with open(file_path, 'r', encoding="utf8") as f:
            srt_text = f.read()
        return list(srt.parse(srt_text))

    @staticmethod
    def __save_srt(file_path, srt_data):
        """
        保存字幕文件
        :param file_path: 字幕文件路径
        :param srt_data: 字幕数据
        :return:
        """
        with open(file_path, 'w', encoding="utf8") as f:
            f.write(srt.compose(srt_data))

    def __get_video_prefer_audio(self, video_meta, prefer_lang=None):
        """
        获取视频的首选音轨，如果有多音轨， 优先指定语言音轨，否则获取默认音轨
        :param video_meta
        :return:
        """
        if type(prefer_lang) == str and prefer_lang:
            prefer_lang = [prefer_lang]

        # 获取首选音轨
        audio_lang = None
        audio_index = None
        audio_stream = filter(lambda x: x.get('codec_type') == 'audio', video_meta.get('streams', []))
        for index, stream in enumerate(audio_stream):
            if not audio_index:
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
            # 获取默认音轨
            if stream.get('disposition', {}).get('default'):
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
            # 获取指定语言音轨
            if prefer_lang and stream.get('tags', {}).get('language') in prefer_lang:
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
                break

        # 如果没有音轨， 则不处理
        if audio_index is None:
            logger.warn(f"没有音轨，不进行处理")
            return False, None, None

        logger.info(f"选中音轨信息：{audio_index}, {audio_lang}")
        return True, audio_index, audio_lang

    def __get_video_prefer_subtitle(self, video_meta, prefer_lang=None):
        """
        获取视频的首选字幕，如果有多字幕， 优先指定语言字幕， 否则获取默认字幕
        :param video_meta:
        :return:
        """
        # from https://wiki.videolan.org/Subtitles_codecs/
        """
        https://trac.ffmpeg.org/wiki/ExtractSubtitles
        ffmpeg -codecs | grep subtitle
         DES... ass                  ASS (Advanced SSA) subtitle (decoders: ssa ass ) (encoders: ssa ass )
         DES... dvb_subtitle         DVB subtitles (decoders: dvbsub ) (encoders: dvbsub )
         DES... dvd_subtitle         DVD subtitles (decoders: dvdsub ) (encoders: dvdsub )
         D.S... hdmv_pgs_subtitle    HDMV Presentation Graphic Stream subtitles (decoders: pgssub )
         ..S... hdmv_text_subtitle   HDMV Text subtitle
         D.S... jacosub              JACOsub subtitle
         D.S... microdvd             MicroDVD subtitle
         D.S... mpl2                 MPL2 subtitle
         D.S... pjs                  PJS (Phoenix Japanimation Society) subtitle
         D.S... realtext             RealText subtitle
         D.S... sami                 SAMI subtitle
         ..S... srt                  SubRip subtitle with embedded timing
         ..S... ssa                  SSA (SubStation Alpha) subtitle
         D.S... stl                  Spruce subtitle format
         DES... subrip               SubRip subtitle (decoders: srt subrip ) (encoders: srt subrip )
         D.S... subviewer            SubViewer subtitle
         D.S... subviewer1           SubViewer v1 subtitle
         D.S... vplayer              VPlayer subtitle
         DES... webvtt               WebVTT subtitle
        """
        image_based_subtitle_codecs = (
            'dvd_subtitle',
            'dvb_subtitle',
            'hdmv_pgs_subtitle',
        )

        if prefer_lang is str and prefer_lang:
            prefer_lang = [prefer_lang]

        # 获取首选字幕
        subtitle_lang = None
        subtitle_index = None
        subtitle_count = 0
        subtitle_stream = filter(lambda x: x.get('codec_type') == 'subtitle', video_meta.get('streams', []))
        for index, stream in enumerate(subtitle_stream):
            # 如果是强制字幕，则跳过
            if stream.get('disposition', {}).get('forced'):
                continue
            # image-based 字幕，跳过
            if (
                    'width' in stream
                    or stream.get('codec_name') in image_based_subtitle_codecs
            ):
                continue
            if not subtitle_index:
                subtitle_index = index
                subtitle_lang = stream.get('tags', {}).get('language')
            # 获取默认字幕
            if stream.get('disposition', {}).get('default'):
                subtitle_index = index
                subtitle_lang = stream.get('tags', {}).get('language')
            # 获取指定语言字幕
            if prefer_lang and stream.get('tags', {}).get('language') in prefer_lang:
                subtitle_index = index
                subtitle_lang = stream.get('tags', {}).get('language')

            subtitle_count += 1

        # 如果没有字幕， 则不处理
        if subtitle_index is None:
            logger.debug(f"没有内嵌字幕")
            return False, None, None, None

        logger.debug(f"命中内嵌字幕信息：{subtitle_index}, {subtitle_lang}")
        return True, subtitle_index, subtitle_lang, subtitle_count

    def __is_noisy_subtitle(self, content):
        """
        判断是否为背景音等字幕
        :param content:
        :return:
        """
        for token in self._noisy_token:
            if content.startswith(token[0]) and content.endswith(token[1]):
                return True
        return False

    def __merge_srt(self, subtitle_data):
        """
        合并整句字幕
        :param subtitle_data:
        :return:
        """
        subtitle_data = copy.deepcopy(subtitle_data)
        # 合并字幕
        merged_subtitle = []
        sentence_end = True

        for index, item in enumerate(subtitle_data):
            # 当前字幕先将多行合并为一行，再去除首尾空格
            content = item.content.replace('\n', ' ').strip()
            # 去除html标签
            parse = etree.HTML(content)
            if parse is not None:
                content = parse.xpath('string(.)')
            if content == '':
                continue
            item.content = content

            # 背景音等字幕，跳过
            if self.__is_noisy_subtitle(content):
                merged_subtitle.append(item)
                sentence_end = True
                continue

            if not merged_subtitle or sentence_end:
                merged_subtitle.append(item)
            elif not sentence_end:
                merged_subtitle[-1].content = f"{merged_subtitle[-1].content} {content}"
                merged_subtitle[-1].end = item.end

            # 如果当前字幕内容以标志符结尾，则设置语句已经终结
            if content.endswith(tuple(self._end_token)):
                sentence_end = True
            # 如果上句字幕超过一定长度，则设置语句已经终结
            elif len(merged_subtitle[-1].content) > 350:
                sentence_end = True
            else:
                sentence_end = False

        return merged_subtitle

    def __do_translate_with_retry(self, text, retry=3):
        # 调用OpenAI翻译
        # 免费OpenAI Api Limit: 20 / minute
        openai = OpenAi(self._openai_key, self._openai_url, self._openai_proxy, self._openai_model)
        ret, result = openai.translate_to_zh(text)
        for i in range(retry):
            if ret and result:
                break
            if "Rate limit reached" in result or "Rate limit exceeded" in result:
                logger.info(f"OpenAI Api Rate limit reached, sleep 60s ...")
                time.sleep(60)
            else:
                logger.warn(f"翻译失败，重试第{i + 1}次")
                time.sleep(2 * (i + 1))
            ret, result = openai.translate_to_zh(text)

        if not ret or not result:
            return None

        return result

    def __translate_zh_subtitle(self, source_lang, source_subtitle, dest_subtitle):
        """
        调用OpenAI 翻译字幕
        :param source_subtitle:
        :param dest_subtitle:
        :return:
        """
        # 读取字幕文件
        srt_data = self.__load_srt(source_subtitle)
        # 合并字幕语句，目前带标点带英文效果较好，非英文或者无标点的需要NLP处理
        if source_lang in ['en', 'eng']:
            logger.info(f"开始合并字幕语句 ...")
            merged_data = self.__merge_srt(srt_data)
            logger.info(f"合并字幕语句完成，合并前字幕数量：{len(srt_data)}, 合并后字幕数量：{len(merged_data)}")
            srt_data = merged_data

        batch = []
        max_batch_tokens = 1000
        for srt_item in srt_data:
            # 跳过空行和无意义的字幕
            if not srt_item.content:
                continue
            if self.__is_noisy_subtitle(srt_item.content):
                continue

            # 批量翻译，减少调用次数
            batch.append(srt_item)
            # 当前批次字符数
            batch_tokens = sum([len(x.content) for x in batch])
            # 如果当前批次字符数小于最大批次字符数，且不是最后一条字幕，则继续
            if batch_tokens < max_batch_tokens and srt_item != srt_data[-1]:
                continue

            batch_content = '\n'.join([x.content for x in batch])
            result = self.__do_translate_with_retry(batch_content)
            # 如果翻译失败，则跳过
            if not result:
                batch = []
                continue

            translated = result.split('\n')
            if len(translated) != len(batch):
                logger.info(
                    f"翻译结果数量不匹配，翻译结果数量：{len(translated)}, 需要翻译数量：{len(batch)}, 退化为单条翻译 ...")
                # 如果翻译结果数量不匹配，则退化为单条翻译
                for index, item in enumerate(batch):
                    result = self.__do_translate_with_retry(item.content)
                    if not result:
                        continue
                    item.content = result + '\n' + item.content
            else:
                logger.debug(f"翻译结果数量匹配，翻译结果数量：{len(translated)}")
                for index, item in enumerate(batch):
                    item.content = translated[index].strip() + '\n' + item.content

            batch = []

        # 保存字幕文件
        self.__save_srt(dest_subtitle, srt_data)

    @staticmethod
    def __external_subtitle_exists(video_file, prefer_langs=None):
        """
        外部字幕文件是否存在
        :param video_file:
        :return:
        """
        video_dir, video_name = os.path.split(video_file)
        video_name, video_ext = os.path.splitext(video_name)

        if type(prefer_langs) == str and prefer_langs:
            prefer_langs = [prefer_langs]

        for subtitle_lang in prefer_langs:
            dest_subtitle = os.path.join(video_dir, f"{video_name}.{subtitle_lang}.srt")
            if os.path.exists(dest_subtitle):
                return True, subtitle_lang

        return False, None

    def __target_subtitle_exists(self, video_file):
        """
        目标字幕文件是否存在
        :param video_file:
        :return:
        """
        if self.translate_zh:
            prefer_langs = ['zh', 'chi']
        else:
            prefer_langs = ['en', 'eng']

        exist, lang = self.__external_subtitle_exists(video_file, prefer_langs)
        if exist:
            return True

        video_meta = Ffmpeg().get_video_metadata(video_file)
        if not video_meta:
            return False
        ret, subtitle_index, subtitle_lang, _ = self.__get_video_prefer_subtitle(video_meta, prefer_lang=prefer_langs)
        if ret and subtitle_lang in prefer_langs:
            return True

        return False

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        """
        拼装插件配置页面，需要返回两块数据：1、页面配置；2、数据结构
        """
        return [
            {
                'component': 'VForm',
                'content': [
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 4
                                },
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'enabled',
                                            'label': '启用插件',
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 4
                                },
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'send_notify',
                                            'label': '发送通知',
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 4
                                },
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'run_now',
                                            'label': '立即运行一次',
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 3
                                },
                                'content': [
                                    {
                                        'component': 'VSelect',
                                        'props': {
                                            'model': 'asr_engine',
                                            'label': 'ASR引擎',
                                            'items': [
                                                {'title': 'faster-whisper', 'value': 'faster-whisper'}
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 3
                                },
                                'content': [
                                    {
                                        'component': 'VSelect',
                                        'props': {
                                            'model': 'faster_whisper_model',
                                            'label': '模型',
                                            'items': [
                                                {'title': 'tiny', 'value': 'tiny'},
                                                {'title': 'tiny.en', 'value': 'tiny.en'},
                                                {'title': 'base', 'value': 'base'},
                                                {'title': 'base.en', 'value': 'base.en'},
                                                {'title': 'small', 'value': 'small'},
                                                {'title': 'small.en', 'value': 'small.en'},
                                                {'title': 'medium', 'value': 'medium'},
                                                {'title': 'large-v1', 'value': 'large-v1'},
                                                {'title': 'large-v2', 'value': 'large-v2'},
                                                {'title': 'large-v3', 'value': 'large-v3'},
                                                {'title': 'distil-small.en', 'value': 'distil-small.en'},
                                                {'title': 'distil-medium.en', 'value': 'distil-medium.en'},
                                                {'title': 'distil-large-v2.en', 'value': 'distil-large-v2'},
                                                {'title': 'distil-large-v3.en', 'value': 'distil-large-v3'},
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 3
                                },
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'translate_zh',
                                            'label': '翻译为中文',
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 3
                                },
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'translate_only',
                                            'label': '仅已有字幕翻译',
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12
                                },
                                'content': [
                                    {
                                        'component': 'VTextarea',
                                        'props': {
                                            'model': 'path_list',
                                            'label': '媒体路径',
                                            'rows': 5,
                                            'placeholder': '要进行字幕生成的路径，每行一个路径，请确保路径正确'
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                },
                                'content': [
                                    {
                                        'component': 'VTextField',
                                        'props': {
                                            'model': 'file_size',
                                            'label': '文件大小（MB）',
                                            'placeholder': '单位 MB, 大于该大小的文件才会进行字幕生成'
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                },
                                'content': [
                                    {
                                        'component': 'VAlert',
                                        'props': {
                                            'type': 'info',
                                            'variant': 'tonal',
                                            'text': '翻译依赖 OpenAi 插件配置'
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ], {
            "enabled": False,
            "send_notify": False,
            "run_now": False,
            "asr_engine": "faster-whisper",
            "faster_whisper_model": "base",
            "translate_zh": True,
            "translate_only": False,
            "path_list": "",
            "file_size": "10",
        }

    def get_api(self) -> List[Dict[str, Any]]:
        pass

    def get_page(self) -> List[dict]:
        pass

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        pass

    def get_state(self) -> bool:
        """
        获取插件状态，如果插件正在运行， 则返回True
        """
        return self._running

    def stop_service(self):
        """
        退出插件
        """
        if self._running:
            self._running = False
            self._scheduler.shutdown()
            logger.info(f"停止自动字幕生成服务")
