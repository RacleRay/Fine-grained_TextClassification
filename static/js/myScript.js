
var dic = {
    0: "这次必须给全五分！因为太完美了！整个就餐体验相当棒！\\n口味：菜品做的都很干净！有图有真相，而且我都是随手拍的！鸡汁萝卜太鲜美了！就是份量有点少，下次我一个人吃一份哈哈哈！牛腱菜花很惊艳，已经吃腻了猪肉的，试试牛肉的非常好吃！带的有孩子还送了小米粥，这里的小米粥熬的很好很够火候！雪菜豆腐也很不错！豆豉蒸芥菜孩子喜欢吃！\\n环境：西湖春天万象城店的环境我很喜欢！虽然开在商场里面，但是感觉很好，并不吵闹，而是很清净。\\n服务：服务员都不错！这次是个男服务员，一直很耐心的推荐！\\n人均：不点螃蟹的话50-80元\\n这家店是每次带孩子来十有八九的选择！",
    1: "在莲花国际广场的地下一楼，店的位置藏得比较深，转了一圈才找到。服务员还是比较热情的，有求必应。先上了四小碟冷菜，泡菜很好吃。价格还是比较适中的，肉类的都差不多二十多块。点了梅花肉、护心肉、猪颈肉、调味牛舌、菌菇拼盘，最喜欢猪颈肉，烤好后够嫩。海鲜汤一般，红红的，有点辣，里面的两只虾都不新鲜。海鲜饼还可以，蘸送的调料吃很棒。另外，生菜叶和火炭都是免费的，最后还送了烤鸡蛋。店里人不多，所以请服务员帮忙烤的。吃好正好用掉两张团购券，满意。平时中午还有半份套餐供应。",
    2: "在新天地看完电影出来逛的时候正好路过这家店，被店门口的精致的蛋糕吸引所以进去一探究竟，店不大，不过装修的很简洁干净，喜欢这个tiffany蓝的主色调，进门就是一个柜台里面陈列了一些蛋糕，难以抉择啊，最后还是挑选了他家的镇店之宝，里面的冰霜或者冰淇淋共六个味道随便选一种，然后服务员说蛋糕是后厨现场做的，这个倒是挺满意的，本来以为就里面拿出来可以吃了，蛋糕上面是酥饼有点甜，感觉一般，中间是冰淇淋，味道就和普通的区别不大，于是乎镇店之宝有点让我们两期望过高而失望了，其实也不至于不好吃，就是没有惊艳到我们而已！75块一块蛋糕么价格是略贵了！",
    3: "这家麦多馅饼在区庄地铁站附近不远，一家店兼卖馅饼和奶茶，奶茶是niconico茶库的，真不敢相信，茶库在时尚天河的店是多么高大上啊，怎么区庄店是跟卖饼的挤在一起，看团购里面显示也没有区庄店了，难道是以前合作过现在不做了，但是确实野还是在卖奶茶，杯子包装跟天河店还是一样\\n的。店很小，就是前面一个柜台，摆着餐牌，后面是一个保温箱，烤箱应该是在后面，点餐后就直接从保温箱里面拿出来。\\n      店主是一男一女两个中年人，态度还不错，一直笑盈盈的，问什么都很愉快的回答，团购是六块五，比现买便宜五毛钱，团购的意义大概就是充一下经验，给霸王餐啥的打基础吧，周六人比较多，卖的比较快，口味不会很全，总会有两三个口味的还在烤制中，要等。\\n      要了一个鸡腿肉的馅饼，馅饼比想象中大，有餐厅吃饭的放碗下面的碟子那么大，厚度也有一两厘米的样子，7元一个也没有想象中性价比那么低，表皮是烤的金黄的，并没有什么油，要了一口，味道一般，不是想象中那种酥脆的口感，而是面饼的那种，毕竟不是炸的煎的，第一口只有吃到面皮，第二口吃到里面的馅料了，但是馅料并不是都是鸡肉，鸡肉还是不多的，就是切成丁的，大部分都是酱和其他东西，面粉含量还是挺高的，馅饼里面并不是中空的，而是像溶洞一样都是空洞，之间塞进少许馅料，整体味道还是可以的，就是馅料太薄，如果再厚一点口感会更加满足。",
    4: "一直很喜欢吃他们家的酸菜鱼。爸妈说很早很早就有这家店里，一直开着，这才让我长大了也能吃到，真好。这家店就在马路边上，凹在里面的，但应该很多人都知道这家店。门面不大，里面也不是很大，生意还是不错的。他们家的酸菜鱼好吃，有小份、中份和大份可以选择，我们一般都是吃中份，人多可以选择大份。鱼肉嫩嫩的，里面的酸菜也好吃，酸菜不是外面买的那种，都是他们家自己腌制的，所以味道好。还会送你两个小菜，花生米和凉拌海带，我喜欢吃凉拌海带。服务员是几个阿姨，也谈不上服务怎么样，一盘鱼端上来我们就自己吃了，也不需要过多的服务了。总体来说还是可以的，经济实惠。"
}

function random_fill(){
    var textra = document.getElementById("Textarea1");
    var idx = Math.floor(Math.random() * 5);

    textra.value = dic[idx];
}