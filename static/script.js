document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const tickerInput = document.getElementById("ticker");
    const periodInput = document.getElementById("period");
    const btnText = document.querySelector(".btn-text");
    const spinner = document.getElementById("loading-spinner");
    const errorMsg = document.getElementById("error-message");
    const predictBtn = document.getElementById("predict-btn");
    const chartBadge = document.getElementById("chart-badge");
    const canvas = document.getElementById("predictionChart");

    let predictionChart = null;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. Setup UI for Loading State
        const ticker = tickerInput.value.trim().toUpperCase();
        const period = periodInput.value;

        if (!ticker) return;

        // Reset previous state
        errorMsg.classList.add("hidden");
        errorMsg.innerText = "";
        chartBadge.textContent = `Predicting...`;
        chartBadge.classList.remove("active");
        
        btnText.textContent = "Processing...";
        spinner.classList.remove("hidden");
        predictBtn.disabled = true;

        try {
            // 2. Fetch Data
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ticker, period })
            });
            
            const data = await response.json();

            if (data.status === 'success') {
                // 3. Render Chart
                chartBadge.textContent = `${ticker} Prediction Active`;
                chartBadge.classList.add("active");
                renderChart(data.dates, data.actual, data.predicted, ticker);
            } else {
                throw new Error(data.message || "Failed to make prediction");
            }
        } catch (error) {
            console.error(error);
            errorMsg.innerText = "Error: " + error.message;
            errorMsg.classList.remove("hidden");
            chartBadge.textContent = "Error occurred";
            chartBadge.classList.remove("active");
        } finally {
            // 4. Reset UI
            btnText.textContent = "Generate Prediction";
            spinner.classList.add("hidden");
            predictBtn.disabled = false;
        }
    });

    function renderChart(dates, actual, predicted, ticker) {
        if (predictionChart) {
            predictionChart.destroy();
        }

        const ctx = canvas.getContext("2d");

        // Glow effects setup for chart
        const actualGradient = ctx.createLinearGradient(0, 0, 0, 400);
        actualGradient.addColorStop(0, "rgba(99, 102, 241, 0.4)");
        actualGradient.addColorStop(1, "rgba(99, 102, 241, 0.0)");

        const predictedGradient = ctx.createLinearGradient(0, 0, 0, 400);
        predictedGradient.addColorStop(0, "rgba(236, 72, 153, 0.4)");
        predictedGradient.addColorStop(1, "rgba(236, 72, 153, 0.0)");

        // Set generic formatting
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "'Outfit', sans-serif";

        predictionChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: dates,
                datasets: [
                    {
                        label: `Actual ${ticker}`,
                        data: actual,
                        borderColor: "#6366f1",
                        backgroundColor: actualGradient,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: `Predicted ${ticker} (LSTM + NLP)`,
                        data: predicted,
                        borderColor: "#ec4899",
                        backgroundColor: predictedGradient,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: "rgba(15, 23, 42, 0.9)",
                        titleColor: "#e2e8f0",
                        bodyColor: "#e2e8f0",
                        borderColor: "rgba(255,255,255,0.1)",
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        boxPadding: 4
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: "rgba(255,255,255,0.05)",
                            drawBorder: false
                        },
                        ticks: {
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        grid: {
                            color: "rgba(255,255,255,0.05)",
                            drawBorder: false
                        },
                        beginAtZero: false
                    }
                }
            }
        });
    }

    // Optional: Draw an empty chart on load just for visual aesthetics
    function renderEmptyChart() {
        const ctx = canvas.getContext("2d");
        Chart.defaults.color = "#94a3b8";
        
        predictionChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                datasets: [{
                    label: "Awaiting Data",
                    data: [100, 105, 102, 110, 108, 115, 112],
                    borderColor: "rgba(255,255,255,0.1)",
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false }
                },
                animation: false
            }
        });
    }

    renderEmptyChart();
});
