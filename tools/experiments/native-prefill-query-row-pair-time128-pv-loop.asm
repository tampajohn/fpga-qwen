
build/native-cpu/prefill-query-row-pair-time128-build/qwen_cpu:	file format mach-o arm64

Disassembly of section __TEXT,__text:

000000010000b928 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl5chunkEPKiiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESA_mmi>:
10000d620: 91020268    	add	x8, x19, #0x80
10000d624: 6b0d011f    	cmp	w8, w13
10000d628: f81603a8    	stur	x8, [x29, #-0xa0]
10000d62c: 1a8db109    	csel	w9, w8, w13, lt
10000d630: 93407d29    	sxtw	x9, w9
10000d634: f81783aa    	stur	x10, [x29, #-0x88]
10000d638: d37ef71c    	lsl	x28, x24, #2
10000d63c: b4000273    	cbz	x19, 0x10000d688 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl5chunkEPKiiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESA_mmi+0x1d60>
10000d640: 3cfc6967    	ldr	q7, [x11, x28]
10000d644: f85803a8    	ldur	x8, [x29, #-0x80]
10000d648: 3cfc6911    	ldr	q17, [x8, x28]
10000d64c: 3cfc6a26    	ldr	q6, [x17, x28]
10000d650: 3cfc6812    	ldr	q18, [x0, x28]
10000d654: 3cfc6985    	ldr	q5, [x12, x28]
10000d658: 3cfc6833    	ldr	q19, [x1, x28]
10000d65c: 3cfc6844    	ldr	q4, [x2, x28]
10000d660: 3cfc6874    	ldr	q20, [x3, x28]
10000d664: 3cfc69c3    	ldr	q3, [x14, x28]
10000d668: 3cfc6895    	ldr	q21, [x4, x28]
10000d66c: 3cfc68a2    	ldr	q2, [x5, x28]
10000d670: 3cfc68d6    	ldr	q22, [x6, x28]
10000d674: 3cfc69e1    	ldr	q1, [x15, x28]
10000d678: 3cfc68f7    	ldr	q23, [x7, x28]
10000d67c: 3cfc6bc0    	ldr	q0, [x30, x28]
10000d680: 3cfc6b30    	ldr	q16, [x25, x28]
10000d684: 14000011    	b	0x10000d6c8 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl5chunkEPKiiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESA_mmi+0x1da0>
10000d688: 6f00e400    	movi.2d	v0, #0000000000000000
10000d68c: 6f00e401    	movi.2d	v1, #0000000000000000
10000d690: 6f00e402    	movi.2d	v2, #0000000000000000
10000d694: 6f00e403    	movi.2d	v3, #0000000000000000
10000d698: 6f00e404    	movi.2d	v4, #0000000000000000
10000d69c: 6f00e405    	movi.2d	v5, #0000000000000000
10000d6a0: 6f00e406    	movi.2d	v6, #0000000000000000
10000d6a4: 6f00e407    	movi.2d	v7, #0000000000000000
10000d6a8: 6f00e411    	movi.2d	v17, #0000000000000000
10000d6ac: 6f00e412    	movi.2d	v18, #0000000000000000
10000d6b0: 6f00e413    	movi.2d	v19, #0000000000000000
10000d6b4: 6f00e414    	movi.2d	v20, #0000000000000000
10000d6b8: 6f00e415    	movi.2d	v21, #0000000000000000
10000d6bc: 6f00e416    	movi.2d	v22, #0000000000000000
10000d6c0: 6f00e417    	movi.2d	v23, #0000000000000000
10000d6c4: 6f00e410    	movi.2d	v16, #0000000000000000
10000d6c8: f940334d    	ldr	x13, [x26, #0x60]
10000d6cc: f85883a8    	ldur	x8, [x29, #-0x78]
10000d6d0: f9400d1b    	ldr	x27, [x8, #0x18]
10000d6d4: b980037b    	ldrsw	x27, [x27]
10000d6d8: aa1a03f0    	mov	x16, x26
10000d6dc: b9800b48    	ldrsw	x8, [x26, #0x8]
10000d6e0: f85903ba    	ldur	x26, [x29, #-0x70]
10000d6e4: aa1b0f5b    	orr	x27, x26, x27, lsl #3
10000d6e8: 9b087f68    	mul	x8, x27, x8
10000d6ec: 8b082148    	add	x8, x10, x8, lsl #8
10000d6f0: 8b0801bb    	add	x27, x13, x8
10000d6f4: aa1303ed    	mov	x13, x19
10000d6f8: bc6d7a98    	ldr	s24, [x20, x13, lsl #2]
10000d6fc: bc6d7ab9    	ldr	s25, [x21, x13, lsl #2]
10000d700: bc6d7ada    	ldr	s26, [x22, x13, lsl #2]
10000d704: acc8737b    	ldp	q27, q28, [x27], #0x100
10000d708: 0e217b7d    	fcvtl	v29.4s, v27.4h
10000d70c: 4f9813a7    	fmla.4s	v7, v29, v24[0]
10000d710: 4f9913a5    	fmla.4s	v5, v29, v25[0]
10000d714: 4f9a13a3    	fmla.4s	v3, v29, v26[0]
10000d718: bc6d7afe    	ldr	s30, [x23, x13, lsl #2]
10000d71c: 4f9e13a1    	fmla.4s	v1, v29, v30[0]
10000d720: 4e217b7b    	fcvtl2	v27.4s, v27.8h
10000d724: 4f981371    	fmla.4s	v17, v27, v24[0]
10000d728: 4f991373    	fmla.4s	v19, v27, v25[0]
10000d72c: 4f9a1375    	fmla.4s	v21, v27, v26[0]
10000d730: 4f9e1377    	fmla.4s	v23, v27, v30[0]
10000d734: 0e217b9b    	fcvtl	v27.4s, v28.4h
10000d738: 4e217b9c    	fcvtl2	v28.4s, v28.8h
10000d73c: 4f981366    	fmla.4s	v6, v27, v24[0]
10000d740: 4f981392    	fmla.4s	v18, v28, v24[0]
10000d744: 4f991364    	fmla.4s	v4, v27, v25[0]
10000d748: 4f991394    	fmla.4s	v20, v28, v25[0]
10000d74c: 4f9a1362    	fmla.4s	v2, v27, v26[0]
10000d750: 4f9a1396    	fmla.4s	v22, v28, v26[0]
10000d754: 4f9e1360    	fmla.4s	v0, v27, v30[0]
10000d758: 4f9e1390    	fmla.4s	v16, v28, v30[0]
10000d75c: 910005ad    	add	x13, x13, #0x1
10000d760: eb0901bf    	cmp	x13, x9
10000d764: 54fffcab    	b.lt	0x10000d6f8 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl5chunkEPKiiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESA_mmi+0x1dd0>
10000d768: 8b1c0168    	add	x8, x11, x28
10000d76c: ad004507    	stp	q7, q17, [x8]
10000d770: ad014906    	stp	q6, q18, [x8, #0x20]
10000d774: 8b1c0188    	add	x8, x12, x28
10000d778: ad004d05    	stp	q5, q19, [x8]
10000d77c: ad015104    	stp	q4, q20, [x8, #0x20]
10000d780: 8b1c01c8    	add	x8, x14, x28
10000d784: ad005503    	stp	q3, q21, [x8]
10000d788: ad015902    	stp	q2, q22, [x8, #0x20]
10000d78c: 8b1c01e8    	add	x8, x15, x28
10000d790: ad005d01    	stp	q1, q23, [x8]
10000d794: 9100814a    	add	x10, x10, #0x20
10000d798: ad014100    	stp	q0, q16, [x8, #0x20]
10000d79c: f101c31f    	cmp	x24, #0x70
10000d7a0: 91004318    	add	x24, x24, #0x10
10000d7a4: aa1003fa    	mov	x26, x16
10000d7a8: 54fff483    	b.lo	0x10000d638 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl5chunkEPKiiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESA_mmi+0x1d10>
10000d7ac: f85783aa    	ldur	x10, [x29, #-0x88]
10000d7b0: 9140214a    	add	x10, x10, #0x8, lsl #12 ; =0x8000
10000d7b4: f85503ad    	ldur	x13, [x29, #-0xb0]
10000d7b8: f85603b3    	ldur	x19, [x29, #-0xa0]
10000d7bc: 6b1301bf    	cmp	w13, w19
10000d7c0: 54fff2ec    	b.gt	0x10000d61c <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl5chunkEPKiiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESA_mmi+0x1cf4>
