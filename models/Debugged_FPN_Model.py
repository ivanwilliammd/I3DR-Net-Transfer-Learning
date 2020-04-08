FPN(
  (C1): Sequential(
    (0): Conv3d(1, 18, kernel_size=(7, 7, 7), stride=(2, 2, 1), padding=(3, 3, 3))
    (1): ReLU(inplace)
  )
  (C2): Sequential(
    (0): MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1, dilation=1, ceil_mode=False)
    (1): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(18, 18, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(18, 18, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(18, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
      (downsample): Conv3d(18, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
    (2): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(72, 18, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(18, 18, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(18, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (3): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(72, 18, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(18, 18, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(18, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
  )
  (C3): Sequential(
    (0): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(72, 36, kernel_size=(1, 1, 1), stride=(2, 2, 2))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(36, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
      (downsample): Conv3d(72, 144, kernel_size=(1, 1, 1), stride=(2, 2, 2))
    )
    (1): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(144, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(36, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (2): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(144, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(36, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (3): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(144, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(36, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
  )
  (C4): Sequential(
    (0): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(144, 72, kernel_size=(1, 1, 1), stride=(2, 2, 2))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(72, 288, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
      (downsample): Conv3d(144, 288, kernel_size=(1, 1, 1), stride=(2, 2, 2))
    )
    (1): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(288, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(72, 288, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (2): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(288, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(72, 288, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (3): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(288, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(72, 288, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (4): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(288, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(72, 288, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (5): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(288, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(72, 288, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
  )
  (C5): Sequential(
    (0): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(288, 144, kernel_size=(1, 1, 1), stride=(2, 2, 2))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(144, 144, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(144, 576, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
      (downsample): Conv3d(288, 576, kernel_size=(1, 1, 1), stride=(2, 2, 2))
    )
    (1): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(576, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(144, 144, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(144, 576, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
    (2): ResBlock(
      (conv1): Sequential(
        (0): Conv3d(576, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv3d(144, 144, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): ReLU(inplace)
      )
      (conv3): Conv3d(144, 576, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      (relu): ReLU(inplace)
    )
  )
  (P1_upsample): Interpolate()
  (P2_upsample): Interpolate()
  (P5_conv1): Conv3d(576, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (P4_conv1): Conv3d(288, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (P3_conv1): Conv3d(144, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (P2_conv1): Conv3d(72, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (P1_conv1): Conv3d(18, 36, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (P1_conv2): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (P2_conv2): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (P3_conv2): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (P4_conv2): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (P5_conv2): Conv3d(36, 36, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
)
