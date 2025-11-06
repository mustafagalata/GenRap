import os
from dotenv import load_dotenv
from lyricsgenius import Genius

load_dotenv()
token = os.environ.get("API_KEY")

if not token:
    raise ValueError("Token cannot found!")

genius = Genius(token, timeout=15, remove_section_headers=True, skip_non_songs=True, retries=3)

artists = ["Rota", "Sansar Salvo", "Defkhan", "Hidra", "Şehinşah", "Ceza",
           "Sagopa Kajmer", "Sayedar", "Massaka", "Killa Hakan", "Cashflow",
           "Dr. Fuchs", "Joker", "Allame", "Tankurt Manas"]

output_dir = "raw_data"

for name in artists:
    try:
        artist = genius.search_artist(name, max_songs=20, sort="popularity")

        if artist is None:
            print("Artist '{}' not found!".format(name))
            continue

        for song in artist.songs:
            clear_song_title = song.title.replace(" ", "_").replace("\\", "_").replace(":", "_")
            filename = f"{artist.name.replace('/', '_')}-{clear_song_title}.txt"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(song.lyrics)

            except Exception as e:
                print(f"{song.title} song couldn't save!: {e}")

    except IOError as e:
        print(f"An error is occured while processing {name}: {e}")


