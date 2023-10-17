# Makefile

# Shell command to check if a directory is non-empty
define check_directory
$(shell [ "$$(ls -A $(1))" ] && echo "not_empty")
endef

# Directory where songs are stored
SONGS_DIR=./songs

# Default all task
all: run-model

run-shell-command:
	@echo "Running shell command"
	sh install.sh

run-downloader: run-shell-command
	@if [ "$(call check_directory,$(SONGS_DIR))" != "not_empty" ]; then \
		echo "Directory is empty, downloading playlists"; \
		python3.10 song_downloader.py; \
	else \
		echo "Data already downloaded"; \
	fi

run-extractor:run-downloader
	@echo "Running extractor"
	python3.10 extractor.py

run-model:run-extractor
	@echo "Running model"
	python3.10 model.py

run-trainer:run-model
	@echo "Running model"
	python3.10 trainer.py