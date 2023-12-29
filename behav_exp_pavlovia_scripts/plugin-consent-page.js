jsPsych.plugins['consent-form'] = (function () {

  var plugin = {};

  plugin.info = {
    name: 'consent-form',
    description: '',
    parameters: {}
  };

  plugin.trial = function(display_element, trial) {

    // Define HTML
    var html = '';

    // Insert CSS.
    
    html += `<style>
      .consent-header {
        text-align: left;
        font-size: 18px;
        font-weight: bold;
        margin-left: 15%;
        margin-right: 15%;
      }
      .consent-warning {
        font-size: 16px;
        font-weight: bold;
        color: red;
      }
      .consent-body {
        text-align: left;
        font-size: 16px;
        margin-bottom: 20px;
        margin-left: 15%;
        margin-right: 15%;
      }
      #jspsych-consent-form-button {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 40px;  /* Adjust space below button */
        padding: 15px 20px;  /* Make button larger */
        cursor: pointer;  /* Change cursor on hover */
      }
    </style>`;

    html += '<h3>Informed Consent Agreement</h3>'
    html += '<p class="consent-warning">Please read this consent agreement carefully. You must be 18 years old or older to participate.</p>'
    html += '<hr>';

    html += '<div id="consent-form">';
    html += '<p class="consent-header">Purpose of the research:<p>';
    html += '<p class="consent-body">To examine how the visual system processes objects and scenes.</p>';
    
    html += '<p class="consent-header">What you will do in this study:<p>';
    html += '<p class="consent-body">You will view displays on a computer monitor and/or listen to sounds via headphones, \
    and respond via a keyboard, response box, mouse or joystick. Stimuli will include letters, words, \
    numbers, keyboard symbols, shapes or pictures of objects, faces, or scenes. Your task will involve \
    detecting or discriminating items or changes in the display and in some tasks, you may be asked \
    to perform two of these tasks at once.</p>';

    html += '<p class="consent-header">Risks:<p>';
    html += '<p class="consent-body">There are no anticipated risks, beyond those encountered in daily life, \
    associated with participating in this study. The task is attention demanding, so you might experience \
    some fatigue as a result of participation.</p>';

    html += '<p class="consent-header">Compensation:<p>';
    html += '<p class="consent-body">The study will take under 30 minutes to complete. You will receive 0.5 course \
    credit (0.5 subject pool hour) for participating in this study. At the end of the study, you will receive \
    an explanation of the study and the hypotheses. We hope that you will learn a little bit about how \
    psychological research is conducted.</p>';

    html += '<p class="consent-header">Voluntary Withdrawal:<p>';
    html += '<p class="consent-body">Your participation in this study is completely voluntary and you may \
    withdraw from the study at any time without penalty. You will not receive Psychology credit unless you \
    complete the experiment or participate in at least 50 minutes. You may skip any questions or procedures, \
    or you may withdraw by informing the research associate that you no longer wish to participate (no \
    questions will be asked). Your decision to participate, decline, or withdraw participation will have \
    no effect on your status at or relationship with the University of Illinois.</p>';

    html += '<p class="consent-header">Confidentiality:<p>';
    html += '<p class="consent-body">Your participation in this study will remain confidential, and \
    your identity will not be stored with your data. Your responses will be assigned a code number that is \
    not linked to your name or other identifying information. All data and consent forms will be stored in a \
    locked room. Results of this study may be presented at conferences and/or published in books, journals, \
    and/or in the popular media. However, if required by laws or University Policy, study information which \
    identifies you and the consent form signed by you may be seen or copied by the following people or \
    groups: 1) The university committee and office that reviews and approves research studies, the \
    Institutional Review Board (IRB) and Office for the Protection of Research Subjects, 2) University \
    and state auditors, and Departments of the university responsible for oversight of research, or 3) \
    Federal government regulatory agencies such as the Office of Human Research Protections in the \
    Department of Health and Human Services.</p>';

    html += '<p class="consent-header">Further information:<p>';
    html += '<p class="consent-body">If you have questions about this study, please contact \
    Dr. Diane Beck, Department of Psychology, University of Illinois, Champaign, IL 61820. \
    Email: dmbeck@illinois.edu; Phone: (217)244-1118.</p>';

    html += '<p class="consent-header">Who to contact about your rights in this study:<p>';
    html += '<p class="consent-body">If you have any concerns about this study or your experience as \
    a participant, you may contact the Institutional Review Board (IRB) at UIUC at (217) 333-2670. \
    Email: irb@uiuc.edu</p>';

    html += '<p class="consent-header">Agreement:<p>';
    html += '<p class="consent-body">The purpose and nature of this research have been sufficiently \
    explained and I agree to participate in this study. I understand that I am free to withdraw at \
    any time without incurring any penalty. I understand that I will receive a copy of this form to \
    take with me.</p>';
    html += '<hr>';

    // Add submit button
    html += '<form id="jspsych-consent-form">';
    html += `<center><input type='submit' id="jspsych-consent-form-button" value="Accept and go to experiment"></input><center>`;
    html += '</form>';

    // Display HTML.
    display_element.innerHTML = html;

    // Define button event listeners
    display_element.querySelector('#jspsych-consent-form').addEventListener('submit', function(event) {

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
