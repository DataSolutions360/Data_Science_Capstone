// document.getElementById("predictButton").onclick = function() { create_gauge_plot() };
// Create the gauge chart.
var gaugePlot = document.getElementById('gauge');

function create_gauge_plot(score) {
    var gaugeData = [{
        domain: { x: [0, 1], y: [0, 1] },
        value: score,
        title: { text: "<b>Heart Health condition</b>" },
        type: "indicator",
        mode: "gauge+number",
        gauge: {
            axis: {
                range: [null, 5],
                tickmode: "array",
                tickvals: [0, 1, 2, 3, 4, 5],
                ticktext: [0, 1, 2, 3, 4, 5]
            },
            bar: { color: "black" },
            steps: [
                { range: [4, 5], color: "red" },
                { range: [3, 4], color: "orange" },
                { range: [2, 3], color: "yellow" },
                { range: [1, 2], color: "lime" },
                { range: [0, 1], color: "green" }
            ]
        }
    }];

    // Create the layout for the gauge chart.
    var gaugeLayout = {
        autosize: true,
        annotations: [{
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            yanchor: 'center',
            text: "The gauge displays your risk level<br>to get a heart disease",
            showarrow: false
        }]
    };
    Plotly.newPlot(gaugePlot, gaugeData, gaugeLayout, { responsive: true });



}

// Function to display inputs after page has reloaded
function repopulate(re_input) {
    console.log(re_input);
    // Check if inputs were entered
    if (!(Object.keys(re_input).length === 0 && re_input.constructor === Object)) {
        document.getElementById("personGender").selectedIndex = re_input.personGender;
        document.getElementById("Age").selectedIndex = re_input.Age;
        document.getElementById("Race").selectedIndex = re_input.Race;
        document.getElementById("Health").selectedIndex = re_input.Health;
        document.getElementById("healthnotgood").selectedIndex = re_input.healthnotgood;
        document.getElementById("BMI").selectedIndex = re_input.BMI;
        document.getElementById("diabetes").selectedIndex = re_input.diabetes;
        document.getElementById("mhealthnotgood").selectedIndex = re_input.mhealthnotgood;
        document.getElementById("drink").selectedIndex = re_input.drink;
        document.getElementById("smoke").selectedIndex = re_input.smoke;
        document.getElementById("asthma").selectedIndex = re_input.asthma;
        document.getElementById("kidney").selectedIndex = re_input.kidney;
        document.getElementById("stroke").selectedIndex = re_input.stroke;
        document.getElementById("cancer").selectedIndex = re_input.cancer;
        document.getElementById("physact").selectedIndex = re_input.physact;
        document.getElementById("sleephours").selectedIndex = re_input.sleephours;

    }

};

document.getElementById("resetButton").onclick = function() { reset() };
// document.getElementById("predictButton").onclick = function() { create_gauge_plot(predict) };

function reset() {
    let zero = 0;
    create_gauge_plot(zero);
    document.getElementById("personGender").selectedIndex = 0;
    document.getElementById("Age").selectedIndex = 1;
    document.getElementById("Race").selectedIndex = 3;
    document.getElementById("Health").selectedIndex = 0;
    document.getElementById("healthnotgood").selectedIndex = 1;
    document.getElementById("BMI").selectedIndex = 1;
    document.getElementById("diabetes").selectedIndex = 0;
    document.getElementById("mhealthnotgood").selectedIndex = 1;
    document.getElementById("drink").selectedIndex = 0;
    document.getElementById("smoke").selectedIndex = 0;
    document.getElementById("asthma").selectedIndex = 0;
    document.getElementById("kidney").selectedIndex = 0;
    document.getElementById("stroke").selectedIndex = 0;
    document.getElementById("cancer").selectedIndex = 1;
    document.getElementById("physact").selectedIndex = 0;
    document.getElementById("sleephours").selectedIndex = 1;                               
};