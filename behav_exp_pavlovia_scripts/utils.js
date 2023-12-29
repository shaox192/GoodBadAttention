

function ratingTrial(propertyName) {
    let msg;

    if (propertyName === "Openness") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - filled) 1----2----3----4----5----6----7 (highest - clear horizon)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`
        
    } else if (propertyName === "Expansion") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - flat) 1----2----3----4----5----6----7 (highest - lines converging at distance)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`

    }  else if (propertyName === "Depth") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - close-up) 1----2----3----4----5----6----7 (highest - panoramic)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`
            
    } else if (propertyName === "Temperature") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - cold) 1----2----3----4----5----6----7 (highest - hot)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`

    } else if (propertyName === "Transience") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - stationary) 1----2----3----4----5----6----7 (highest - transient)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`
            
    } else if (propertyName === "Concealment") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - no where to hide) 1----2----3----4----5----6----7 (highest - lots of hiding spots)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`
        
    } else if (propertyName === "Navigability") {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - cannot move in there) 1----2----3----4----5----6----7 (highest - go wherever I can)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`
    } else {
        msg = `<div class="prompt">
                <p>To what degree do you think this image is in terms of <span class="highlight">${propertyName}</span>?</p>
                <p>(lowest - few things) 1----2----3----4----5----6----7 (highest - crowded)</p>
                <p>Please use the keyboard number keys from 1 to 7 to respond.</p>
            </div>`
    }

    return {
        type: "image-keyboard-response",
        stimulus: jsPsych.timelineVariable('stimulus'),
        choices: ['1', '2', '3', '4', '5', '6', '7'],
        data: jsPsych.timelineVariable('data'),
        stimulus_height: 640,
        prompt: msg
    };
}

function transitionInstruction(propertyName) {
    let msg;

    if (propertyName === "openness") {
        msg = `<div class="rating_transition"> 
                <p>In the next block, you will be asked to rate for <span class="highlight">Openness</span></p>
                <p><span class="highlight">Openness</span> is how open or closed the scene looks. </p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): the view is filled with things like walls, surfaces, objects, and textures</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): you can see a clear horizon with nothing blocking the view</p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
               </div>`
    } else if (propertyName === "expansion") {
        msg = `<div class="rating_transition"> 
                <p>In the next block, you will be asked to rate for <span class="highlight">Expansion</span></p>
                <p><span class="highlight">Expansion</span> is how much the scene seems to stretch out in front of you. </p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): the scene looks flat</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): there are parallel lines that seem to meet at a point at far distance</p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    }  else if (propertyName === "depth") {
        msg = `<div class="rating_transition"> 
                <p>In the next block, you will be asked to rate for <span class="highlight">Depth</span></p>
                <p><span class="highlight">Depth</span> is the scale or size of the space, </p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): a close-up view on single surfaces or objects</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): a wide view of panoramic scenes</p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    } else if (propertyName === "temperature") {
        msg = `<div class="rating_transition"> 
                <p>In the next block, you will be asked to rate for <span class="highlight">Temperature</span></p>
                <p><span class="highlight">Temperature</span> is how hot or cold you'd feel if you were physically in the scene.</p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): cold</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): hot</p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    } else if (propertyName === "transience") {
        msg = `<div class="rating_transition"> 
                <p>In the next block, you will be asked to rate for <span class="highlight">Transience</span></p>
                <p><span class="highlight">Transience</span> is how quickly things in the scene are changing. 
                Changing could mean something in the image is moving like running water, or the overall scene changing, like when it depicts a sunset.</p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest):  seem stationary</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): the scene depends
                on the photograph being taken at that exact moment </p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    } else if (propertyName === "concealment") {
        msg = `<div class="rating_transition"> 
                <p>In the next block, you will be asked to rate for <span class="highlight">Concealment</span></p>
                <p><span class="highlight">Concealment</span> is how easy it would be for people to hide in the scene OR 
                how many elements are hidden in the photo. </p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): the scene is out in the open with nowhere to hide</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): it has lots of hiding spots because of many objects and surfaces </p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    } else if (propertyName === "navigability") {
        msg = `<div class="rating_transition"><p>In the next block, you will be asked to rate for <span class="highlight">Navigability</span></p>
                <p><span class="highlight">Navigability</span> is how easy it would be to move around in the scene. </p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): it is filled with obstacles making it hard to move OR an environment too dangerous to move</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): this is an open space where you can go in any direction easily </p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    } else {
        msg = `<div class="rating_transition"><p>In the next block, you will be asked to rate for <span class="highlight">Clutter</span></p>
                <p><span class="highlight">Clutter</span> is how filled or crowded the scene is. </p>
                <p>You will rate on a scale from 1 to 7, where: </p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;1 (lowest): it's simple with few items or objects</p>
                <p>&nbsp;&nbsp;&nbsp;&nbsp;7 (highest): it's packed with many things, making it look busy or messy</p>
                <p>Feel free to take a break. If you are ready, please press SPACEBAR to continue </p>
              </div>`
    }

    return {
        type: "html-keyboard-response",
        stimulus: msg,
        choices: ['SPACEBAR']
    };
}