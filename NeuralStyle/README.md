# 图像风格迁移
@[toc]
## 简介
- 利用卷积神经网络实现图像风格的迁移。
## 画风迁移
- 简单来说就是将另一张图像的绘画风格在不改变原图图像内容的情况下加入到原图像中，从而“创造”名家风格的绘画作品。
- 这其中牵扯到很多难题，但是很多问题已经被一一攻克，一般而言，这类问题将涉及到图像风格的捕捉、图像风格的迁移、图像风格的组合等主要流程。
- 整体看来，卷积核特征的捕捉及应用、图像相干性的优化、画风的组合将是难题。
## 图像风格捕捉
- 原理
  - 通过将格拉姆矩阵（gram matrix）应用于卷积神经网络各层能够捕获该层的样式，所以，如果从填充了随机噪声的图像开始，对其进行优化使得网络各层的格拉姆矩阵与目标图像的格拉姆矩阵相匹配，那么不难理解，生成的图像将会模仿目标图像的风格。
  - 可以定义一个style损失函数，计算两组激活输出值各自减去格拉姆矩阵之后计算平方误差。在原始图像和目标图像（莫奈的《睡莲》）上进行训练，将两幅图片输入VGG16的卷积神经网络，对每个卷积层计算上述style损失并多层累加，对损失使用lbfgs进行优化（它需要梯度值和损失值进行优化）。
  - 其实，很多观点认为，匹配图像风格的最好方式是直接匹配所有层的激活值，事实上，格拉姆矩阵效果会更好一些，这点并不容易理解。其背后的原理是，通过计算给定层的每个激活值与其他激活值的乘积，我们获得了神经元之间的相关性。这些相关性可以理解为图像风格的编码，因为他们衡量了激活值的分布情况，而不是激活值本身。
  - 这也带来了几个问题。一个就是零值问题，在任一被乘数为零的情况下，一个向量和自己的转置的点积都会是零，模型无法在零值处识别相关性。由于零值频繁出现，可以在执行点积前为特征值添加一个小的差量delta，delta去-1即可。还有一个问题就是，我们计算了所有激活值的格拉姆矩阵，难道不是应该针对像素通道计算吗？事实上，我们为每个像素的通道都计算了格拉姆矩阵，然后观察它们在整个图像上的相关性，这样做提供了一个捷径：可以计算通道均值并将其用作格拉姆矩阵，这会帮助获得一幅平均风格的图像，更有普适性。
  - 同时，添加了总变分损失（total variation loss），要求网络时刻检查相邻像素的差异，否则，图像会趋于像素化且更加不平缓。某种程度上，这种方法与用于持续检查层权重与层输出的正则化过程非常类似，整体效果相当于在输出像素上添加了一个略微模糊的滤镜。（本质上，这是一种模糊化的方法）该部分的结果是将最后一个组成成分添加到了损失函数中，使得图像整体更像内容图像而不是风格图像。这里所做的，就是有效优化生成图像，使得上层的激活值对应图像内容，下层的激活值对应图像风格，也就是网络底层对应图像风格，网络高层对应图像内容，通过这种方式实现图像风格转换。
- 代码
  - ```python
    def gram_matrix(x):
        if K.image_data_format() != 'channels_first':
            x = K.permute_dimensions(x, (2, 0, 1))
        features = K.batch_flatten(x)
        return K.dot(features - 1, K.transpose(features - 1)) - 1

    def style_loss(layer_1, layer_2):
        gr1 = gram_matrix(layer_1)
        gr2 = gram_matrix(layer_2)
        return K.sum(K.square(gr1 - gr2)) / (np.prod(layer_2.shape).value ** 2)

    w, h = 740, 468
    style_image = K.variable(preprocess_image(style1_image_path, target_size=(h, w)))
    result_image = K.placeholder(style_image.shape)
    input_tensor = K.concatenate([style_image, result_image], axis=0)
    print(input_tensor.shape)

    model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # caculate loss
    feature_outputs = [layer.output for layer in model.layers if '_conv' in layer.name]
    loss_style = K.variable(0.)

    for idx, layer_features in enumerate(feature_outputs):
        loss_style += style_loss(layer_features[0, :, :, :], layer_features[1, :, :, :])

    class Evaluator(object):
        def __init__(self, loss_total, result_image, **other):
            grads = K.gradients(loss_total, result_image)
            outputs = [loss_total] + list(other.values()) + grads
            self.iterate = K.function([result_image], outputs)
            self.other = list(other.keys())
            self.other_values = {}
            self.shape = result_image.shape

            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            outs = self.iterate([x.reshape(self.shape)])
            self.loss_value = outs[0]
            self.grad_values = outs[-1].flatten().astype('float64')
            self.other_values = dict(zip(self.other, outs[1:-1]))
            return self.loss_value

        def grads(self, x):
            return np.copy(self.grad_values)
        

    style_evaluator = Evaluator(loss_style, result_image)

    def run(evaluator, image, num_iter=50):
        for i in range(num_iter):
            clear_output()
            image, min_val, info = fmin_l_bfgs_b(evaluator.loss, image.flatten(), fprime=evaluator.grads, maxfun=20)
            showarray(deprocess_image(image.copy(), h, w))
            print("Current loss value:", min_val)
            print(' '.join(k + ':' + str(evaluator.other_values[k]) for k in evaluator.other))
        return image

    x = np.random.uniform(0, 255, result_image.shape) - 128.
    res = run(style_evaluator, x, num_iter=50)

    def total_variation_loss(x, exp=1.25):
        _, d1, d2, d3 = x.shape
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :d2 - 1, :d3 - 1] - x[:, :, 1:, :d3 - 1])
            b = K.square(x[:, :, :d2 - 1, :d3 - 1] - x[:, :, :d2 - 1, 1:])
        else:
            a = K.square(x[:, :d1 - 1, :d2 - 1, :] - x[:, 1:, :d2 - 1, :])
            b = K.square(x[:, :d1 - 1, :d2 - 1, :] - x[:, :d1 - 1, 1:, :])
        return K.sum(K.pow(a + b, exp))

    loss_variation = total_variation_loss(result_image) / 5000
    loss_with_variation = loss_variation + loss_style
    evaluator_with_variation = Evaluator(loss_with_variation, result_image)

    x = np.random.uniform(0, 255, result_image.shape) - 128.
    res = run(evaluator_with_variation, x, num_iter=100)
    ```
- 结果
  - ![](https://img-blog.csdnimg.cn/20190730195156361.png)
  - ![](https://img-blog.csdnimg.cn/20190730212119588.png)
  - 显然，使用总变分损失后，原来的噪声图像产生的结果更像一个内容图像了。
## 图像风格迁移
- 原理
  - 要想将捕获到的风格从一个图像应用到另一个图像上，需要使用一个损失函数来平衡一个图像的内容和另一个图像的风格。
  - 在现有图像上运行上面的代码是不难的，但是结果并不那么令人满意，它看起来似乎将风格应用到了现有图像上，但是程序的不断执行，原始图像不断分解，坚持运行，最后生成一幅与原图独立的新图像，显然，这不是想要。可以通过在损失函数中添加第三个组成成分来解决这个问题，该成分会考虑生成图像和参考图像的差异，这就形成了内容损失（content loss），其作用于网络最后一层。最后一层包含与网络所看到的的映像最近似的内容，因此这也是真正希望保持一致的内容。 
  - 通过把各层在网络中的位置纳入考量来微调风格损失（style loss），希望较低层承载更多权重，因为较低层能够捕获更多的图像纹理/风格，也希望较高层更多地参与到图像内容的捕获之中。这样，算法更容易平衡图像内容（使用最后一层）和图像风格（主要较低层）。最后，平衡损失函数的三个部分。
- 代码
  - ```python
    def content_loss(base, combination):
        return K.sum(K.square(combination - base))

    w, h = load_img(base_image_path).size
    base_image = K.variable(preprocess_image(base_image_path))
    style_image = K.variable(preprocess_image(style2_image_path, target_size=(h, w)))
    combination_image = K.placeholder(style_image.shape)
    input_tensor = K.concatenate([base_image, style_image, combination_image], axis=0)

    model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    feature_outputs = [layer.output for layer in model.layers if '_conv' in layer.name]
    loss_content = content_loss(feature_outputs[-1][0, :, :, :], feature_outputs[-1][2, :, :, :])
    loss_variation = total_variation_loss(combination_image)
    loss_style = K.variable(0.)

    for idx, layer_features in enumerate(feature_outputs):
        loss_style += style_loss(layer_features[1, :, :, :], layer_features[2, :, :, :]) * (0.5 ** idx)

    loss_content /= 40
    loss_variation /= 10000
    loss_total = loss_content + loss_variation + loss_style

    combined_evaluator = Evaluator(loss_total, combination_image, loss_content=loss_content, loss_variation=loss_variation, loss_style=loss_style)
    run(combined_evaluator, preprocess_image(base_image_path), num_iter=100)
    ```
- 结果
  - 素材
    - ![原图](https://img-blog.csdnimg.cn/20190730215802994.png)
    - ![风格图](https://img-blog.csdnimg.cn/20190730215854499.png)
  - 通过梵高的《星空》作为风格图，将其风格迁移到阿姆斯特丹的老教堂图片上，效果如下（这是风格迁移很经典的图）。
    - ![](https://img-blog.csdnimg.cn/20190730221451928.png)
## 图像风格内插
- 原理
  - 已经捕获了两种图像风格，希望在另一幅图像上应用一种介于两者之间的风格，如何做呢？可以使用一个包含浮点数值的损失函数，浮点数值表示每种风格应用的百分比。
  - 加载两种不同风格的图像，为每种风格创建损失值，然后引入占位符summerness，调控百分比来调整各种风格的占比。
  - 通过在损失变量中再次增加成分，使得可以指定两种不同风格的权重，当然，也可以进一步增加更多的风格图像并调整权重，但是要注意下调那些“压制”效果好的风格比重，以免其影响较大。
- 代码
  - ```python
    w, h = load_img(base_image_path).size
    base_image = K.variable(preprocess_image(base_image_path))
    winter_style_image = K.variable(preprocess_image('data/road-to-versailles-at-louveciennes.jpg', target_size=(h, w)))
    summer_style_image = K.variable(preprocess_image('data/VanGogh_Farmhouse.jpeg', target_size=(h, w)))
    combination_image = K.placeholder(summer_style_image.shape)
    input_tensor = K.concatenate([base_image, summer_style_image, winter_style_image, combination_image], axis=0)

    model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    feature_outputs = [layer.output for layer in model.layers if '_conv' in layer.name]
    loss_content = content_loss(feature_outputs[-1][0, :, :, :], feature_outputs[-1][2, :, :, :])
    loss_variation = total_variation_loss(combination_image)

    loss_style_summer = K.variable(0.)
    loss_style_winter = K.variable(0.)
    for idx, layer_features in enumerate(feature_outputs):
        loss_style_summer += style_loss(layer_features[1, :, :, :], layer_features[-1, :, :, :]) * (0.5 ** idx)
        loss_style_winter += style_loss(layer_features[2, :, :, :], layer_features[-1, :, :, :]) * (0.5 ** idx)

    loss_content /= 40
    loss_variation /= 10000

    summerness = K.placeholder()
    loss_total = (loss_content + loss_variation + loss_style_summer * summerness + loss_style_winter * (1 - summerness))

    combined_evaluator = Evaluator(loss_total, 
                                combination_image, 
                                loss_content=loss_content, 
                                loss_variation=loss_variation, 
                                loss_style_summer=loss_style_summer,
                                loss_style_winter=loss_style_winter)
    iterate = K.function([combination_image, summerness], combined_evaluator.iterate.outputs)
    combined_evaluator.iterate = lambda inputs: iterate(inputs + [1.0])  # 1.0夏天风格
    res = run(combined_evaluator, preprocess_image(base_image_path), num_iter=50)

    path = 'data/summer_winter_%d/20.jpg'
    def save(res, step):
        img = deprocess_image(res.copy(), h, w)
        imsave(path % step, img)

    for step in range(1, 21, 1):
        combined_evaluator = Evaluator(loss_total, 
                                    combination_image, 
                                    loss_content=loss_content, 
                                    loss_variation=loss_variation, 
                                    loss_style_summer=loss_style_summer,
                                    loss_style_winter=loss_style_winter)
        iterate = K.function([combination_image, summerness], combined_evaluator.iterate.outputs)
        combined_evaluator.iterate = lambda inputs: iterate(inputs + [1.0 - step / 20.])  # 0.05-1.0的夏天风格的图像
        res = run(combined_evaluator, preprocess_image(base_image_path), num_iter=50)
        save(res, step)
    ```
- 结果
  - 素材
    - ![夏天](https://img-blog.csdnimg.cn/20190730221917272.png)
    - ![冬天](https://img-blog.csdnimg.cn/20190730222149740.png)
  - 通过调控上述两个风格的比例，设定summer风格100%，结果如下。
    - ![](https://img-blog.csdnimg.cn/20190730223319277.png)
  - 通过循环调控夏天风格的比例，产生多幅图像，组合形成动态图，如下。
    - ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93czEuc2luYWltZy5jbi9sYXJnZS8wMDZvbVV2bmx5MWc1aTl6YTdjZmFnMzBrazBkMHF2Yi5naWY)
## 补充说明
- 参考书籍《深度学习实战：Deep Learning Cookbook》。
- 具体项目代码上传至[我的Github](https://github.com/luanshiyinyang/DeepLearningProject/tree/master/NeuralStyle)，欢迎Star或者Fork。
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。