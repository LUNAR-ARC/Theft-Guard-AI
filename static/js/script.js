function pollIntel() {
    fetch('/get_intel')
        .then(res => res.json())
        .then(data => {
            const status = document.getElementById('status');
            const log = document.getElementById('intel-log');

            // Sir, if danger is detected, we make it VERY obvious
            if (data.danger) {
                document.body.style.backgroundColor = "#2d0a10"; // Dark red pulse
                status.innerText = "!! THREAT DETECTED !!";
                status.style.color = "#f7768e";
            } else {
                document.body.style.backgroundColor = "#0d0d12";
                status.innerText = "SYSTEM_SECURE";
                status.style.color = "#bb9af7";
            }

            // Sir, this ensures every message from the AI is logged instantly
            if (data.intel && data.intel !== lastMessage && data.intel !== "Scanning patterns...") {
                const entry = document.createElement('div');
                entry.className = data.danger ? 'log-entry danger-entry' : 'log-entry';
                entry.innerHTML = `<span class="time">[${new Date().toLocaleTimeString()}]</span> ${data.intel}`;
                log.prepend(entry);
                lastMessage = data.intel;
            }
        });
}