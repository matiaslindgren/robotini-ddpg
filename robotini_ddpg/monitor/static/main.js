function limitPrecision(_, x) {
	if (typeof x === 'number') {
		return parseFloat(x.toFixed(4));
	}
	return x;
}

function updateTeamContainer(teamId, allData) {
	const teamDiv = document.getElementById(teamId);
	const {original_frame, observation_x, observation_y, ...metadata} = allData;
	teamDiv.querySelector("pre.metadata").innerHTML = JSON.stringify(metadata, limitPrecision, 2);
	teamDiv.querySelector(".images img.originalFrame").src = original_frame;
	teamDiv.querySelector(".images img.observationX").src = observation_x;
	teamDiv.querySelector(".images img.observationY").src = observation_y;
}


function readAllChunks(readableStream, teamId) {
	const reader = readableStream.getReader();
	let chunks = [];

	function pump() {
		return reader.read().then(({ value, done }) => {
			const decoded = new TextDecoder("utf-8").decode(value);
			try {
				const parsed = JSON.parse(chunks.join("") + decoded);
				updateTeamContainer(teamId, parsed);
				chunks = [];
			} catch {
				if (chunks.length < 50) {
					chunks.push(decoded);
				} else {
					chunks = [];
				}
			}
			if (done) {
				console.log("End of stream, reconnecting...", teamId);
				connect(teamId);
				return;
			}
			return pump();
		});
	}

	return pump();
}


function connect(teamId, n) {
	const maxNumConnectAttempts = 100;
	n = n || 0;
	console.log("Connecting", teamId);
	if (n >= maxNumConnectAttempts) {
		console.error("Unable to connect to server, tried", n, "times");
		return;
	}
	fetch("/stream/" + teamId)
		.then(response => readAllChunks(response.body, teamId))
		.catch(err => {
			console.warn("Unable to connect to server. Reconnecting...", n)
			setTimeout(connect, (n+1) * 500, teamId, n+1);
		})
}


for (let teamId of teamIds) {
	connect(teamId);
}
