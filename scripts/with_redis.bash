#!/usr/bin/env bash
set -ue

redis_pid=''

function cleanup() {
	if [ ! -z $redis_pid ]; then
		echo "cleanup pid '$redis_pid'"
		kill -s TERM $redis_pid
		redis_pid=''
	fi
}

trap cleanup SIGINT SIGTERM SIGQUIT EXIT

redis-server \
	--unixsocket $REDIS_SOCKET \
	--save "" \
	--maxmemory 100mb \
	--port 0 > redis.log 2>&1 &
redis_pid=$!

$@

cleanup
