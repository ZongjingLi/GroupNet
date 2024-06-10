var root = "outputs/"; 
var idx = 1;
var split = "val"
var name_ = "expr"
var base_path = "outputs/" + name_ + "/"
var frame_num = 1;

function toNextBatch() {
    idx += 1;
    updateData();
}

function toPrevBatch() {
    if (idx > 1) {
        idx -= 1;
        updateData();
    }
}

function toggleNextFrame() {
    frame_num += 1;
    if (frame_num > 2) {
        frame_num = 1;
    }
    updateData()
}

function updateData() {
    var img_bind = document.getElementById("input-image");
    var img_path = base_path + split +"/" + idx + "_img"+frame_num+".png";
    img_bind.src = img_path 

    var img_bind = document.getElementById("motion-strength");
    var img_path = base_path + split +"/" + idx + "_motion.png";
    img_bind.src = img_path 

    var img_bind = document.getElementById("predict-mask");
    var img_path = base_path + split +"/" + idx + "_predict_mask.png";
    img_bind.src = img_path 

}