// document.getElementById("predictButton").onclick = function() { create_gauge_plot() };
// Create the gauge chart.
var gaugePlot = document.getElementById('gauge');

function create_gauge_plot(score) {
    console.log(score)
    var gaugeData = [{
        domain: { x: [0, 1], y: [0, 1] },
        value: score,
        title: { text: "<b>Heart Health Predictor</b>" },
        type: "indicator",
        mode: "gauge+number",
        // mode: "gauge",
        gauge: {
            axis: {
                range: [null, 100],
                tickmode: "array",
                tickwidth: 1,
                tickcolor: "Black"
            },
            bar: { color: "black" },
            steps: [
                { range: [0, 20], color: "blue" },
                { range: [20, 40], color: "green" },
                { range: [40, 60], color: "yellow" },
                { range: [60, 80], color: "orange" },
                { range: [80, 100], color: "red" }]
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
            y: 0,
            yanchor: 'center',
            text: " ",
            showarrow: false
        }]
    };
    Plotly.newPlot("gauge", gaugeData, gaugeLayout, { responsive: true });}

// Function to display inputs after page has reloaded
function repopulate(re_input) {
    console.log("here");
    console.log(re_input);
    // Check if inputs were entered
    if (!(Object.keys(re_input).length === 0 && re_input.constructor === Object)) {
        document.getElementById("personGender").value = re_input.Birth_Sex;
        document.getElementById("Age").value = re_input.Age;
        document.getElementById("Race").value = re_input.Race;
        document.getElementById("Health").value = re_input.Overall_Health;
        document.getElementById("healthnotgood").value = re_input.Physical_Health;
        document.getElementById("BMI").value = re_input.BMI_CDC_Categories;
        document.getElementById("diabetes").value = re_input.Diabetes;
        document.getElementById("mhealthnotgood").value = re_input.Mental_Health;
        document.getElementById("drink").value = re_input.Alcohol_Usage;
        document.getElementById("smoke").value = re_input.Tobacco_Usage;
        document.getElementById("asthma").value = re_input.Asthma_History;
        document.getElementById("kidney").value = re_input.Kidney_Disease;
        document.getElementById("stroke").value = re_input.Stroke;
        document.getElementById("cancer").value = re_input.Colonoscopy;
        document.getElementById("physact").value = re_input.Physical_Activity;
        document.getElementById("sleephours").value = re_input.Avg_Hours_of_Sleep;
        console.log(re_input.Birth_Sex, re_input.Age,re_input.Race, re_input.Overall_Health,re_input.Physical_Health, re_input.BMI_CDC_Categories,re_input.Diabetes);
        console.log(re_input.Mental_Health, re_input.Alcohol_Usage,re_input.Tobacco_Usage, re_input.Asthma_History,re_input.Kidney_Disease, re_input.Stroke,re_input.Colonoscopy);
        console.log(re_input.Physical_Activity, re_input.Avg_Hours_of_Sleep);
    }

};

document.getElementById("resetButton").onclick = function() { reset() };
// document.getElementById("predictButton").onclick = function() { create_gauge_plot(predict) };

function reset() {
    let zero = 0;
    create_gauge_plot(zero);
    document.getElementById("personGender").value = 0;
    document.getElementById("Age").value = 1;
    document.getElementById("Race").value = 1;
    document.getElementById("Health").value = 0;
    document.getElementById("healthnotgood").value = 1;
    document.getElementById("BMI").value = 1;
    document.getElementById("diabetes").value = 1;
    document.getElementById("mhealthnotgood").value = 1;
    document.getElementById("drink").value = 0;
    document.getElementById("smoke").value = 0;
    document.getElementById("asthma").value = 0;
    document.getElementById("kidney").value = 0;
    document.getElementById("stroke").value = 0;
    document.getElementById("cancer").value = 1;
    document.getElementById("physact").value = 0;
    document.getElementById("sleephours").value = 1;  
                         
};