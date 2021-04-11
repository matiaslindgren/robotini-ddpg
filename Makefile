# PYTHON_BIN := /usr/local/opt/python@3.9/bin/python3
# SIMULATOR_HOST := localhost

PYTHON_BIN := python3.9
SIMULATOR_HOST := 192.168.1.211

SIMULATOR_LOG_URL := $(SIMULATOR_HOST):11001
SIMULATOR_URL := $(SIMULATOR_HOST):11000
REDIS_SOCKET := /tmp/robotini-ddpg.redis.sock
CACHE_DIR := ./tf-cache
MONITORED_TEAM_IDS := test_snail{1..3}

TF_CPP_MIN_LOG_LEVEL := 1

export TF_CPP_MIN_LOG_LEVEL

.PHONY: start_agent_monitor env_demo train_tf_ddpg start_redis start_tensorboard clean

start_monitor:
	$(PYTHON_BIN) robotini_ddpg/monitor/webapp.py $(REDIS_SOCKET) $(MONITORED_TEAM_IDS)

env_demo:
	$(PYTHON_BIN) tests/test_env.py $(SIMULATOR_URL) $(SIMULATOR_LOG_URL) $(REDIS_SOCKET)

train_tf_ddpg:
	$(PYTHON_BIN) tf_agent_train.py

start_redis:
	redis-server --unixsocket $(REDIS_SOCKET) --save "" --maxmemory 100mb --port 0

start_tensorboard:
	tensorboard --logdir $(CACHE_DIR)

clean:
	redis-cli -s $(REDIS_SOCKET) flushall
	rm -rv $(CACHE_DIR)
