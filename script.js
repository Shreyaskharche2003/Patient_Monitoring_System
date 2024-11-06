function runEyeDetection() {
    fetch('/eye')
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok: " + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('eye-output').innerText = data.result || data.error;
        })
        .catch(error => {
            document.getElementById('eye-output').innerText = "Error: " + error;
        });
}


function runMaskDetection() {
    fetch('/mask')
        .then(response => response.json())
        .then(data => {
            document.getElementById('mask-output').innerText = data.result || data.error;
        })
        .catch(error => {
            document.getElementById('mask-output').innerText = "Error: " + error;
        });
}

function storeMotionData() {
    const motionData = { motion: "walking" }; // Example data; modify as needed
    fetch('/motion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(motionData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('motion-output').innerText = data.message || data.error;
    })
    .catch(error => {
        document.getElementById('motion-output').innerText = "Error: " + error;
    });
}
