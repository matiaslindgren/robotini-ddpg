import argparse
import json
import sys
import time

from flask import Flask, render_template, Response
from redis import Redis

from robotini_ddpg.util import sleep_until


def run(fps_limit, redis_socket_path, monitored_team_ids, host, port):
    redis = Redis(unix_socket_path=redis_socket_path)
    app = Flask(__name__)

    def generate_stream(team_id):
        next_frame = time.perf_counter()
        while True:
            next_frame += 1/fps_limit
            state_json = redis.hget(team_id, "state_snapshot.json")
            if state_json:
                yield state_json + b"\r\n"
            sleep_until(next_frame)

    @app.route("/")
    def index():
        return render_template("index.html", team_ids=monitored_team_ids)

    @app.route("/stream/<team_id>")
    def stream(team_id):
        return Response(generate_stream(team_id), mimetype="application/json")

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps-limit", type=int, default=20)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("redis_socket_path", type=str)
    parser.add_argument("monitored_team_ids", nargs="+")
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
