import argparse
import json
import logging
import sys
import time

from flask import Flask, render_template, Response
from redis import Redis
from redis.exceptions import ConnectionError

from robotini_ddpg.util import sleep_until


def run(fps_limit, redis_socket_path, monitored_team_ids, host, port):
    redis = Redis(unix_socket_path=redis_socket_path)
    app = Flask(__name__)

    def generate_stream():
        state = len(monitored_team_ids)*[b'{}']
        next_frame = time.perf_counter()
        while True:
            next_frame += 1/fps_limit
            for i, team_id in enumerate(monitored_team_ids):
                state[i] = redis.hget(team_id, "state_snapshot.json") or b'{}'
            yield b'[' + b','.join(state) + b']' + b'\r\n'
            sleep_until(next_frame)

    @app.route("/")
    def index():
        return render_template("index.html", team_ids=monitored_team_ids)

    @app.route("/team-ids")
    def team_ids():
        return {"teamIds": monitored_team_ids}

    @app.route("/stream")
    def stream():
        return Response(generate_stream(), mimetype="application/json")

    app.logger.setLevel(logging.INFO)
    app.logger.info("monitored ids: '%s'", ' '.join(monitored_team_ids))
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
