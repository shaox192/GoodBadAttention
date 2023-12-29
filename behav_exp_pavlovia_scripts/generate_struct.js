

function generateImgSeq(img_ls) {
    const imgIdx = Array.from({ length: img_ls.length }, (_, i) => i);
    imgIdx.sort(() => Math.random() - 0.5); // Simple shuffle

    const randomizedImgFiles = imgIdx.map(i => img_ls[i]);

    const interval = 10;
    const finalImgFiles = [];
    for (let j = 0; j < randomizedImgFiles.length; j += interval) {
        const currTrials = randomizedImgFiles.slice(j, j + interval);
        finalImgFiles.push(...currTrials);
        const repTr = currTrials[Math.floor(Math.random() * currTrials.length)];
        finalImgFiles.push(repTr);
    }

    return finalImgFiles;
}

function generateStruct(properties_ls, img_dir, num_uniq_img) {

    const data = {
        blocks: [],
        images: {}
    };

    const properties_idx = Array.from({ length: properties_ls.length }, (_, i) => i).sort(() => Math.random() - 0.5);
    const properties_rand_ls = [];
    for (const i of properties_idx) {
        properties_rand_ls.push(properties_ls[i])
    }
    data.blocks = properties_rand_ls

    // img_dir = "./images/exp"
    // const exp_num_img_uniq = 80;
    const img_ls = [];
    for (i = 0; i < num_uniq_img; i++){
        img_ls.push(img_dir + "/im" + i + ".jpg")
    };  

    for (const p of properties_ls) {
        data.images[p] = generateImgSeq(img_ls);
    }

    // You can now use the 'data' object in your experiment as needed
    console.log(data);
    
    return data;
}
