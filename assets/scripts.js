var root = "outputs/"; 
var idx = 0;
var split = "train"
var name_ = "expr"
var base_path = "outputs/" + name_ + "/"

var sample_idx = 0;
var dataset_name = "TDWKitchen"
var dataset_split = "train"
var dataset_path = "/Users/melkor/Documents/datasets/"


// this part handles the data change of the model ouptut
function setName() {
    name_ = document.getElementById("name_input").value;
    changeData(); 
}

function setIdx() {
    idx = document.getElementById("idx_input").value;
    changeData();
}
function changeCurr() {
    document.getElementById("curr_idx").textContent = "current idx: " + idx;
}
    var img_bind = document.getElementById("im_vs_gt");
    var img_path = base_path + split +"/" + idx + "_img.png";
    img_bind.src = img_path //String.prototype.replace("/Users/melkor/Documents/datasets/TDWRoom/test/img/img_test_" + idx + ".png");

    var mask_bind = document.getElementById("masks");
    var mask_path = base_path + split +"/" + idx + "_mask.png";
    mask_bind.src = mask_path
}
function changeQuery(json) {
    var bind = document.getElementById("curr_metric")
    bind.innerText = "Current IoU:"+json.iou;

    bind = document.getElementById("query")
    bind.innerHTML = "<div>"
    for (i = 0; i < json.programs.length; i++) {
        bind.innerHTML += "<div>query:"+json.queries[i]+"</div>"
        bind.innerHTML += "<div>program:"+json.programs[i]+"</div>"
        bind.innerHTML += "<div>answer:"+json.answers[i]+"</div>"
        bind.innerHTML += "<div>gt_answer:"+json.gt_answers[i]+"</div><div>_____________________</div>"
    }
    bind.innerHTML += "</div>"
    console.log()
}
function changeOverall(json) {
    var bind = document.getElementById("overall_metric")
    bind.innerText = "Overall mIoU:"+json.miou + "Overall Acc:" + json.accuracy;
}
function changeData() {
    base_path = "outputs/" + name_ + "/"
    changeImg();
    
    json_path = base_path + split +"/" + idx + "_eval.json";
    fetch(json_path)
    .then((response) => response.json())
    .then((json) => changeQuery(json));

    overall_path = base_path + split +"/" + "overall.json";
    fetch(overall_path)
    .then((response) => response.json())
    .then((json) => changeOverall(json));
    changeCurr();
}
function next() {
    idx = parseInt(idx) + 1;
    changeData();
}
function prev() {
    if (idx > 0) {
        idx = parseInt(idx) - 1;
    }
    changeData();
}

// change the dataset display and choice

function setDatasetName() {
    dataset_name = document.getElementById("choose-dataset-name").value;
    changeDataset(); 
    console.log(dataset_name)
}

function setSampleSplit() {
    dataset_split = document.getElementById("choose-sample-split").value;
    changeDataset();
}

function changeDatasetImg() {
    var img_bind = document.getElementById("gt-ims");
    var img_path = dataset_path + dataset_name + "/" + dataset_split +"/" + "img/img_" + sample_idx + ".png";
    img_bind.src = img_path //String.prototype.replace("/Users/melkor/Documents/datasets/TDWRoom/test/img/img_test_" + idx + ".png");
    //img_bind.src = "Users/melkor/Documents/datasets/TDWKitchen/train/img/img_0.png";
                    //Users/melkor/Documents/datasets/TDWKitchen/train/img/img_0.png 
    var mask_bind = document.getElementById("gt-ids");
    var mask_path = dataset_path + dataset_name + "/" + dataset_split +"/" + "img/id_"+ sample_idx + ".png";
    mask_bind.src = mask_path
    //console.log(img_path)
    //console.log(mask_path)
}
function nextSample() {
    sample_idx = parseInt(sample_idx) + 1;
    changeDataset();
}
function prevSample() {
    if (sample_idx > 0) {
        sample_idx = parseInt(sample_idx) - 1;
    }
    changeDataset();
}
function changeDataset() {
    changeDatasetImg();
}

changeData();
changeDataset();
