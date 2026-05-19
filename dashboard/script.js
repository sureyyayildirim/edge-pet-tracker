// marker
const pet = document.getElementById("petMarker");

// son feeding zamanı saklanacak
let lastFeedingActivity = "-";


// oda koordinatları
const roomPositions = {

    "Kitchen":{
        top:"120px",
        left:"100px"
    },


    "Living Room":{
        top:"120px",
        left:"260px"
    },

    "Bedroom":{
        top:"340px",
        left:"325px"
    },

    "Feeding Area":{
        top:"150px",
        left:"380px"
    }

};


// isim düzenleme
function formatLocationName(location){

    if(location==="living_room")
        return "Living Room";

    if(location==="kitchen")
        return "Kitchen";

    if(location==="bedroom")
        return "Bedroom";

    if(location==="feeding_area")
        return "Feeding Area";

    return location;

}


// saat alma
function getCurrentTime(){

    return new Date().toLocaleTimeString(
        "tr-TR",
        {
            hour:"2-digit",
            minute:"2-digit"
        }
    );

}


// dashboard güncelleme
function updateDashboard(data){

    let currentLocation=
    formatLocationName(
        data.stable_prediction
    );

    let markerLocation=
    formatLocationName(
        data.stable_prediction
    );

    let behaviorStatus=
    "Normal";


    if(data.stable_prediction==="feeding_area"){

        currentLocation="Living Room";

        // varsayılan marker salon

        markerLocation="Living Room";

        const feedingConfirmed =
        data.feeding_confirmed === true ||
        data.feeding_confirmed === "true";


        if(feedingConfirmed){

            behaviorStatus="Feeding";

            // sadece feeding sırasında mama alanına git

            markerLocation="Feeding Area";

            lastFeedingActivity=
            getCurrentTime();

        }

    }


    pet.style.top=
    roomPositions[markerLocation].top;

    pet.style.left=
    roomPositions[markerLocation].left;


    document.getElementById(
        "location"
    ).innerText=
    currentLocation;


    document.getElementById(
        "behavior"
    ).innerText=
    behaviorStatus;


    document.getElementById(
        "feedingTime"
    ).innerText=
    data.feeding_time_24h_min
    +" min";


    document.getElementById(
        "lastFeeding"
    ).innerText=
    lastFeedingActivity;

    document.getElementById(
        "confidence"
    ).innerText =
    data.confidence + "%";

    document.getElementById(
        "anomaly"
    ).innerText=
    data.anomaly;


    document.getElementById(
        "update"
    ).innerText=
    new Date().toLocaleTimeString(
        "tr-TR"
    );

}


// {// TEST
// const testSequence = [
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },
//     { stable_prediction:"feeding_area", feeding_confirmed:true, feeding_time_24h_min:1, anomaly:"normal", confidence:93 },
//     { stable_prediction:"feeding_area", feeding_confirmed:true, feeding_time_24h_min:2, anomaly:"normal", confidence:94 },
//     { stable_prediction:"feeding_area", feeding_confirmed:true, feeding_time_24h_min:3, anomaly:"normal", confidence:95 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:3, anomaly:"normal", confidence:92 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:3, anomaly:"normal", confidence:91 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"kitchen", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"kitchen", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"kitchen", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"kitchen", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"bedroom", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"bedroom", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"bedroom", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"bedroom", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"bedroom", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"bedroom", feeding_confirmed:true, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },   
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"living_room", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"living_room", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"living_room", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    
//     { stable_prediction:"living_room", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:90 },
//     { stable_prediction:"living_room", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },
//     { stable_prediction:"feeding_area", feeding_confirmed:false, feeding_time_24h_min:0, anomaly:"normal", confidence:91 },    

// ];

// let index = 0;

// setInterval(() => {

//     updateDashboard(testSequence[index]);

//     console.log(
//         "Test:",
//         testSequence[index],
//         "Last Feeding Activity:",
//         lastFeedingActivity
//     );

//     index++;

//     if(index >= testSequence.length){
//         index = 0;
//     }

// }, 3000);

// console.log("script.js çalıştı");
//}

const brokerUrl = "wss://broker.hivemq.com:8884/mqtt";
const topic = "pettracker/data";

const client = mqtt.connect(brokerUrl);

client.on("connect", function () {
    console.log("MQTT connected");

    client.subscribe(topic, function (err) {
        if (!err) {
            console.log("Subscribed to:", topic);
        }
    });
});

client.on("message", function (topic, message) {
    console.log("MQTT message:", message.toString());

    const data = JSON.parse(message.toString());

    updateDashboard(data);
});