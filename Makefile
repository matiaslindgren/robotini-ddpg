# PYTHON_BIN := /usr/local/opt/python@3.9/bin/python3
# SIMULATOR_HOST := localhost

PYTHON_BIN := python3.9
SIMULATOR_HOST := 192.168.1.211

LOG_SOCKET_URL := $(SIMULATOR_HOST):11001
CAR_SOCKET_URL := $(SIMULATOR_HOST):11000
REDIS_SOCKET := /tmp/robotini-ddpg.redis.sock
CACHE_DIR := ./tf-cache
MONITORED_TEAM_IDS := test_snail{1..3}
CONFIG_PATH := ./scripts/config.yml

TF_CPP_MIN_LOG_LEVEL := 1

export TF_CPP_MIN_LOG_LEVEL PYTHON_BIN REDIS_SOCKET

.PHONY: clean start_monitor env_demo train_ddpg start_tensorboard

clean:
	rm -rv $(CACHE_DIR)

start_redis:
	redis-server \
		--unixsocket $(REDIS_SOCKET) \
		--save "" \
		--maxmemory 32mb \
		--port 0

train_ddpg:
	$(PYTHON_BIN) scripts/train_ddpg.py \
		--car-socket-url $(CAR_SOCKET_URL) \
		--log-socket-url $(LOG_SOCKET_URL) \
		--redis-socket-path $(REDIS_SOCKET) \
		--config-path $(CONFIG_PATH) \
		--cache-dir $(CACHE_DIR)

start_tensorboard:
	tensorboard --logdir $(CACHE_DIR)
