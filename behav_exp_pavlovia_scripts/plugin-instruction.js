jsPsych.plugins['instruction'] = (function () {

  var plugin = {};

  plugin.info = {
    name: 'instruction',
    description: '',
    parameters: {}
  };

  plugin.trial = function(display_element, trial) {

    // Define HTML
    var html = '';

    // Insert CSS.
    
    html += `<style>
      .header {
        text-align: left;
        font-size: 24px;
        font-weight: bold;
        margin-left: 15%;
        margin-right: 15%;
      }
      .body {
        text-align: left;
        font-size: 20px;
        margin-bottom: 20px;
        margin-left: 15%;
        margin-right: 15%;
      }
      #jspsych-instruction-button {
        font-size: 24px;
        font-weight: bold;
        // margin-bottom: 40px;  /* Adjust space below button */
        padding: 15px 20px;  /* Make button larger */
        cursor: pointer;  /* Change cursor on hover */
      }
      .highlight {
        color: red;
        font-weight: bold;
        font-style: italic;
      }
    </style>`;

    html += '<p class="header">Thank you for completing the pre-experiment steps! Now we are going to go over your task for today:</p>';
    html += '<hr>';

    html += '<p class="body">We are studying how people perceive photographs of scenes. \
    Your task is to evaluate a series of images based on eight key properties: Openness, Expansion, Depth, \
    Temperature, Transience, Concealment, Navigability and Clutter, using a rating scale from 1 to 7; where 1 signifies the lowest and 7 the \
    highest level of each property. You will be rating each property in separate blocks and will \
    have the opportunity to take breaks between them.</p>';
    html += '<p class="body">Below we give a detailed explanation for these properties. They may seem a lot, but no need to worry! You will be \
    reminded of these details again during the experiment: </p>';

    html += '<p class="body"><span class="highlight">Openness</span> is how open or closed the scene looks. \
    At one end, you can see a clear horizon with nothing blocking the view (7:highest). \
    At the other end, the view is filled with things like walls, surfaces, objects, and textures (1: lowest). </p>';

    html += '<p class="body"><span class="highlight">Expansion</span> is how much the scene seems to stretch out in front of you. \
    It can look flat (1: lowest), or it can have parallel lines that seem to meet at a point at far distance (7:highest).</p>';

    html += '<p class="body"><span class="highlight">Depth</span> is the scale or size of the space, \
    ranging from a close-up view on single surfaces or objects (1: lowest) to \
    a wide view of panoramic scenes (7: highest) </p>';

    html += '<p class="body"><span class="highlight">Temperature</span> is how hot (7: highest) \
    or cold (1: lowest) you would feel if you were immersed in the scene. </p>';

    html += '<p class="body"><span class="highlight">Transience</span> is how quickly things in the scene are changing. \
    Changing could mean something in the image is moving like running water, or the overall scene changing, \
    like when it depicts a sunset. At one extreme, the scene could seem stationary (1: lowest), and \
    at the other extreme, the scene could depend on the photograph being taken at that exact moment (7: highest). </p>';

    html += '<p class="body"><span class="highlight">Concealment</span> is how easy it would be for people to \
    hide in the scene OR how many elements are hidden in the photo. It can range from being out in the \
    open with nowhere to hide (1: lowest), to a place with lots of hiding spots because of \
    many objects and surfaces (7: highest).</p>';

    html += '<p class="body"><span class="highlight">Navigability</span> is how easy it would be to move around in the scene. \
    It can range from a place filled with obstacles making it hard to move OR an environment too dangerous to move (1: lowest),\
    to an open space where you can go in any direction easily (7: highest). </p>';

    html += '<p class="body"><span class="highlight">Clutter</span> is how filled or crowded the scene is. \
    At one end, it is simple with few items or objects (1: lowest). \
    At the other end, it is packed with many things, making it look busy or messy (7: highest). </p>';


    html += '<hr>';


    // Add submit button
    html += '<form id="jspsych-instruction">';
    html += `<center><input type='submit' id="jspsych-instruction-button" value="Start experiment"></input><center>`;
    html += '</form>';

    // Display HTML.
    display_element.innerHTML = html;

    // Define button event listeners
    display_element.querySelector('#jspsych-instruction').addEventListener('submit', function(event) {

      // Wait for response
      event.preventDefault();

      // Update screen
      display_element.innerHTML = '';

      // Move onto next trial

      jsPsych.finishTrial();

    });

  };

  return plugin;

})();
