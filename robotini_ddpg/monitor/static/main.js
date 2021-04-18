function limitPrecision(_, x) {
	if (typeof x === 'number') {
		return parseFloat(x.toFixed(4));
	}
	return x;
}

function updateTeamContainer(allData) {
	for (let teamData of allData) {
		if (Object.keys(teamData).length > 0) {
			const {original_frame, observation_x, observation_y, team_id, ...metadata} = teamData;
			const teamDiv = document.getElementById(team_id);
			teamDiv.querySelector("pre.metadata").innerHTML = JSON.stringify(metadata, limitPrecision, 2);
			teamDiv.querySelector(".images img.originalFrame").src = original_frame;
			teamDiv.querySelector(".images img.observationX").src = observation_x;
			teamDiv.querySelector(".images img.observationY").src = observation_y;
		}
	}
}


function readAllChunks(readableStream, teamIds) {
	const reader = readableStream.getReader();
	let chunks = [];

	function pump() {
		return reader.read().then(({ value, done }) => {
			const decoded = new TextDecoder("utf-8").decode(value);
			try {
				const parsed = JSON.parse(chunks.join('') + decoded);
				updateTeamContainer(parsed);
				chunks = [];
			} catch {
				chunks.push(decoded);
				console.log(chunks.length, "chunks");
			}
			if (done) {
				console.log("End of stream, reconnecting...");
				setTimeout(connect, 10, teamIds);
				return;
			}
			return pump();
		});
	}

	return pump();
}


function connect(teamIds, n) {
	const maxNumConnectAttempts = 10;
	n = n || 0;
	console.log("Connecting", teamIds.join(' '));
	if (n >= maxNumConnectAttempts) {
		console.error("Unable to connect to server, tried", n, "times");
		return;
	}
	fetch("/stream")
		.then(response => readAllChunks(response.body, teamIds))
		.catch(err => {
			console.warn("Unable to connect to server, reconnecting", n);
			setTimeout(connect, 1000, teamIds, n+1);
		})
}

window.addEventListener('DOMContentLoaded', _ => {
	fetch("/team-ids")
		.then(response => response.json())
		.then(data => connect(data.teamIds))
});
