// Create the gauge chart.
var gaugePlot = document.getElementById('gauge');

function create_gauge_plot(score) {
    var gaugeData = [{
        domain: { x: [0, 1], y: [0, 1] },
        value: score,
        title: { text: "<b>Death Risk</b>" },
        type: "indicator",
        mode: "gauge+number",
        gauge: {
            axis: {
                range: [null, 5],
                tickmode: "array",
                tickvals: [0, 1, 2, 3, 4, 5],
                ticktext: [0, 1, 2, 3, 4, 5]
            },
            bar: { color: "#808080" },
            steps: [
                { range: [4, 5], color: "#A8534B" },
                { range: [3, 4], color: "#EC9D75" },
                { range: [2, 3], color: "#DCE6A7" },
                { range: [1, 2], color: "#9ADFB0" },
                { range: [0, 1], color: "#57CBAB" }
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
            text: "The gauge displays your risk level<br>of dying from COVID-19. <br> 0-1 = very low risk <br> 1-2 = low risk <br> 2-3 = moderate risk <br> 3-4 = moderate-high risk <br> 4-5 = high risk",
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
        document.getElementById(re_input.sex).checked = true;
        document.getElementById(re_input.state).checked = true;
        document.getElementById(re_input.age_range).checked = true;
        document.getElementById(re_input.race).checked = true;
        document.getElementById(re_input.ethnicity).checked = true;
    }
};

document.getElementById("resetButton").onclick = function() { reset() };

function reset() {
    let zero = 0;
    create_gauge_plot(zero);
    let sex = document.getElementsByName("sex");
    genders.forEach((gender) => { gender.checked = false; });
    let state = document.getElementsByName("state");
    genders.forEach((gender) => { gender.checked = false; });
    let age_range = document.getElementsByName("age_range");
    makes.forEach((make) => { make.checked = false; });
    let race = document.getElementsByName("race");
    bodies.forEach((body) => { body.checked = false; });
    let ethnicity = document.getElementsByName("ethnicity");
    days.forEach((day) => { day.checked = false; });
};
