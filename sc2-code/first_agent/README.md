# 第一个rule-base的星际二agent

## 什么是rule-base agent
简单地讲就是基于规则，在事先想好各种情况，然后面对各种情况下给出对应的做法。rule-base的做法常见于专家系统，在专家系统中，由专家给出对应的知识，然后将知识编码成规则。

可以简单地理解为代码中的条件结构，对每种情况找到对应的条件，然后执行符合条件的代码。在实际实现中，可以简单地用if-elif-else来实现。

rule-base面对复杂的问题，可能会出现很多情况，使的工作量变得巨大。另外就是面对没有符合条件的状况，agent的表现可能会非常差，甚至崩溃。

## 什么是星际二

### 在wiki上的解释为：
《星际争霸II》是一款即时战略游戏，通过俯瞰的视角模式观阅整个战场并对玩家的军队下达指令，最终目标就是击败战场上的对手们。游戏中单人剧情战役主要是扮演人类的角色来进行故事，但玩家可以在多人对战模式中选择《星际争霸II》中的三个独特种族：星灵、人类以及异虫。
每个种族都拥有独特的单位；这些单位都在战场上各自扮演着自己特定的角色。配合不同的单位来组成一支多元化的军队将带领走上胜利之路。
在玩家指挥下最基本的单位便是工作单位，太空工程车、探测机、工虫。它们将采集用来扩展玩家的基地、发展单位和部队所需的资源，同时它们也可用来建造建筑。更进阶的单位将于玩家的基地满足特地需求后开放，例如建造某个建筑或研发某种科技。
在多人对战中，通过歼灭敌人所有的建筑或对方先行投降来取得胜利。

### 我们能做些什么？
在这里我们会采用deepmind的pysc2作为开发环境。之后我们可能会涉及到小地图的战役，资源收集等等星际二中的常见环境。此外我们可能还会采用星际二的地图编辑器进行自定义地图的编写，自己设置触发器来构造特定的学习问题。

在这里我们不涉及到安装的对应内容，如果安装pysc2与可能遇到的问题，请在github上搜索pysc2来获得。

## 我们的rule-base agent

我们的rule-base agent实现了什么功能：
1. 确定自己在地图上位置
2. 选择农民
3. 建立供给站（人族）
4. 建立兵营，设置结合点
5. 无脑冲脸战术

### 基本的pysc2介绍
在pysc2的agents中提供了一个基础的类供我们编写agent时候继承，然后overwrite相应的方法即可

```
class BaseAgent(object):
  """A base agent to write custom scripted agents."""

  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = None
    self.action_spec = None

  def setup(self, obs_spec, action_spec):
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def reset(self):
    self.episodes += 1

  def step(self, obs):
    self.steps += 1
    self.reward += obs.reward
    return actions.FunctionCall(0, [])
```

从方法名即可清楚地知道每个方法的作用是什么, \_\_init\_\_是初始化， setup是设置基本信息， reset是一局结束后调用的方法。重点是step，为每一步调用的函数，用来决定agent做什么。

我们的第一个agent就是通过继承BaseAgent来overwrite step方法实现第一个rule-base的agent，具体效果见视频: [first rule-base agent](https://github.com/wwxFromTju/sc2-101-zh/blob/master/sc2-data/first_agent.mov)，点击一下view raw即可下载观看，差不多9M左右。

### 确定自己在地图上位置
在玩星际，魔兽，红警的时候，地图通常是非常大的，我们只能看到地图上的一部分，如何知道自己在地图上的那里，很重要的就是借助小地图来知道自己在地图上的那个方位。rule-base agent也是如此。这里我们采用的64 * 64的双人地图，利用先验知识：玩家要不出生在左上角，要不就是出生在右下角。

对于人，看一眼就能记住自己在左上角还是右下角，rule-base agent我们需要用一个变量来记住自己是在左上角落还是右下角，此外pysc2中提供了对应的视图内容的信息，如图所示：![图](https://raw.githubusercontent.com/wwxFromTju/sc2-101-zh/master/sc2-data/real_screen.png)

即将人类看到的彩色画面变成等价的二维的数组，对于同一个物体用一堆数字来表示，如图中值为45的3*3的数组，在实际画面中实际为一个scv，降低了开发难度。因为实际图片中的物体具有不同颜色，边缘，并不是一组相同数字，所以具体开发更难，需要自己对图片提取语义。

所以在这里，我们可以通过小地图上的信息来判断agent在那里，与自己相关的单位信息为1：

```
obs.observation["minimap"][_PLAYER_RELATIVE]
```
那么我们通过判断值为1的位置来获得自己单位的坐标，然后获得对应的坐标值， 自己的单位为1：
```
(obs.observation["minimap"][_PLAYER_RELATIVE] ==1).nonzero()
```
通过对自己所有单位的坐标取平均来判断自己的位置，我们是一个64*64的地图，所以判断y坐标小于31（因为是从0开始）就是左上，反之右下：

```
 self.base_top_left = player_y.mean() <= 31
```
### 选择农民
判断完自己的位置后，就是选择scv来建造对应的建筑，选中单位的操作并不是对小地图来进行，而是对于当前的视图来进行的，所以这边类似小地图，来获取当前视图的信息，scv的值为45， 然后来获得第一个scv的坐标:

```
unit_y, unit_x = (unit_type ==45).nonzero()
target = [unit_x[0], unit_y[0]]
```
有了坐标，我们可以选择对应坐标的单位：

```
actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
```
### 建立供给站（人族）
选择完scv后，我们需要判断当前的资源是不是可以建造供给站，在pysc2中可以直接判断对应的操作是不是在当前可行的操作中：

```
_BUILD_SUPPLYDEPOT in obs.observation['available_actions']
```
如果可行的话，我们将供给站建在基地旁边， 所以首先找出基地的坐标，然后给个偏移值，作为供给站建立的位置：

```
unit_type = obs.observation['screen'][_UNIT_TYPE]
unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
target = self.transformLocation(int(unit_x.mean()), 0 , int(unit_y.mean()), 20)
```
如果可以建立供给站，那么就建立供给站：

```
actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_SCREEN, target])
```
### 建立兵营，设置结合点
建完供给站后，就是建立兵营，类似建立供给站，判断资源够不够，是不是可以建：

```
_BUILD_BARRACKS in obs.observation['available_actions']
```
然后建立在基地旁边：

```
actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])
```
建立好兵营就是设置集合点，这边同样是利用先验知识，如果我们在左上，对手就是在右下，反之亦然：

```
if self.base_top_left: actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_MINIMAP, [40, 40]])
else: actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_MINIMAP, [20, 20]])
```

### 无脑冲脸战术
设立好集合点之后，就是无脑冲锋啦，就是类似人类一直点生产枪兵，然后枪兵一直往对方基地送死：

```
actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
```

## 完整代码
[我的代码](https://github.com/wwxFromTju/sc2-101-zh/blob/master/sc2-code/first_agent/first_agent.py)

同时参考了：
   1. [简介与code](https://github.com/wwxFromTju/sc2-101-zh/tree/master/sc2-code/first_agent)
   2. [视频](https://github.com/wwxFromTju/sc2-101-zh/tree/master/sc2-data)

