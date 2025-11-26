import os
import yt_dlp
import whisper
from transformers import pipeline
import tempfile
import re

class YouTubeSummaryApp:
    def __init__(self):
        # Загружаем модель для транскрибации
        self.model = whisper.load_model("base")
        
        # Загружаем модель для суммаризации
        self.summarizer = pipeline(
            "summarization",
            model="IlyaGusev/mbart_ru_sum_gazeta",
            tokenizer="IlyaGusev/mbart_ru_sum_gazeta"
        )
    
    def download_audio(self, youtube_url):
        """Скачиваем аудио с YouTube"""
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            audio_file = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return audio_file, info['title']
    
    def transcribe_audio(self, audio_path):
        """Транскрибируем аудио в текст"""
        result = self.model.transcribe(audio_path)
        return result["text"]
    
    def generate_summary(self, text, num_points=10):
        """Генерируем ключевые мысли"""
        # Очищаем текст
        text = self.clean_text(text)
        
        # Разбиваем текст на части (если слишком длинный)
        chunks = self.split_text(text)
        
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) > 50:  # Суммаризируем только достаточно длинные фрагменты
                summary = self.summarizer(
                    chunk,
                    max_length=100,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
        
        # Если суммаризация не сработала, используем ключевые предложения
        if len(summaries) < num_points:
            key_sentences = self.extract_key_sentences(text, num_points)
            return key_sentences
        
        return summaries[:num_points]
    
    def clean_text(self, text):
        """Очищаем текст от лишних символов"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        return text.strip()
    
    def split_text(self, text, max_length=1000):
        """Разбиваем текст на части"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_key_sentences(self, text, num_sentences=10):
        """Извлекаем ключевые предложения (простой подход)"""
        sentences = text.split('.')
        # Сортируем предложения по длине (как простой показатель важности)
        key_sentences = sorted(
            [s.strip() for s in sentences if len(s.strip()) > 20],
            key=len,
            reverse=True
        )[:num_sentences]
        
        return [s for s in key_sentences if s]
    
    def process_video(self, youtube_url):
        """Основной метод обработки видео"""
        try:
            print("📥 Скачиваем аудио...")
            audio_file, title = self.download_audio(youtube_url)
            
            print("🎙️ Транскрибируем аудио...")
            text = self.transcribe_audio(audio_file)
            
            print("🧠 Анализируем текст...")
            key_points = self.generate_summary(text)
            
            # Удаляем временный аудиофайл
            if os.path.exists(audio_file):
                os.remove(audio_file)
            
            return {
                "title": title,
                "transcription": text,
                "key_points": key_points
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    app = YouTubeSummaryApp()
    
    print("🎯 YouTube Summary Generator")
    print("=" * 40)
    
    youtube_url = input("Введите URL YouTube видео: ")
    
    result = app.process_video(youtube_url)
    
    if "error" in result:
        print(f"❌ Ошибка: {result['error']}")
        return
    
    # Сохраняем результаты в файл
    filename = f"{result['title'].replace(' ', '_')}_summary.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Название: {result['title']}\n")
        f.write("=" * 50 + "\n\n")
        f.write("🔑 Ключевые мысли:\n\n")
        
        for i, point in enumerate(result['key_points'], 1):
            f.write(f"{i}. {point}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("📝 Полный текст транскрибации:\n\n")
        f.write(result['transcription'])
    
    print(f"\n✅ Готово! Результаты сохранены в файл: {filename}")
    print(f"\n🎯 Ключевые мысли ({len(result['key_points']}):")
    print("-" * 40)
    
    for i, point in enumerate(result['key_points'], 1):
        print(f"{i}. {point}")

if __name__ == "__main__":
    main()
