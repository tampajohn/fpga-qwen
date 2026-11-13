
build/native-cpu/head-rowpack4-v1-build/qwen_cpu:	file format mach-o arm64

Disassembly of section __TEXT,__text:

000000010000d1ac <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi>:
10000d580: 91001110    	add	x16, x8, #0x4
10000d584: eb02021f    	cmp	x16, x2
10000d588: 540000e9    	b.ls	0x10000d5a4 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x3f8>
10000d58c: aa0803e9    	mov	x9, x8
10000d590: 17ffff14    	b	0x10000d1e0 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x34>
10000d594: f100415f    	cmp	x10, #0x10
10000d598: 54000aa2    	b.hs	0x10000d6ec <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x540>
10000d59c: d280000b    	mov	x11, #0x0               ; =0
10000d5a0: 14000060    	b	0x10000d720 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x574>
10000d5a4: b940054b    	ldr	w11, [x10, #0x4]
10000d5a8: f9400c0a    	ldr	x10, [x0, #0x18]
10000d5ac: f940014a    	ldr	x10, [x10]
10000d5b0: 7100057f    	cmp	w11, #0x1
10000d5b4: 54000d0b    	b.lt	0x10000d754 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x5a8>
10000d5b8: f940100c    	ldr	x12, [x0, #0x20]
10000d5bc: f940018c    	ldr	x12, [x12]
10000d5c0: f9401d29    	ldr	x9, [x9, #0x38]
10000d5c4: 93407d6e    	sxtw	x14, w11
10000d5c8: 9100418c    	add	x12, x12, #0x10
10000d5cc: 9b0e7d0d    	mul	x13, x8, x14
10000d5d0: 8b0d0529    	add	x9, x9, x13, lsl #1
10000d5d4: 9100812d    	add	x13, x9, #0x20
10000d5d8: d37df1ce    	lsl	x14, x14, #3
10000d5dc: d280000f    	mov	x15, #0x0               ; =0
10000d5e0: 6f00e400    	movi.2d	v0, #0000000000000000
10000d5e4: 6f00e401    	movi.2d	v1, #0000000000000000
10000d5e8: aa1003e9    	mov	x9, x16
10000d5ec: aa0d03f0    	mov	x16, x13
10000d5f0: aa0c03f1    	mov	x17, x12
10000d5f4: 6f00e402    	movi.2d	v2, #0000000000000000
10000d5f8: 6f00e403    	movi.2d	v3, #0000000000000000
10000d5fc: 6f00e404    	movi.2d	v4, #0000000000000000
10000d600: 6f00e405    	movi.2d	v5, #0000000000000000
10000d604: 6f00e406    	movi.2d	v6, #0000000000000000
10000d608: 6f00e407    	movi.2d	v7, #0000000000000000
10000d60c: ad7fc630    	ldp	q16, q17, [x17, #-0x10]
10000d610: ad7f4e12    	ldp	q18, q19, [x16, #-0x20]
10000d614: acc25614    	ldp	q20, q21, [x16], #0x40
10000d618: 0e217a56    	fcvtl	v22.4s, v18.4h
10000d61c: 4e30cec0    	fmla.4s	v0, v22, v16
10000d620: 4e217a52    	fcvtl2	v18.4s, v18.8h
10000d624: 4e31ce41    	fmla.4s	v1, v18, v17
10000d628: 0e217a72    	fcvtl	v18.4s, v19.4h
10000d62c: 4e30ce42    	fmla.4s	v2, v18, v16
10000d630: 4e217a72    	fcvtl2	v18.4s, v19.8h
10000d634: 4e31ce43    	fmla.4s	v3, v18, v17
10000d638: 0e217a92    	fcvtl	v18.4s, v20.4h
10000d63c: 4e30ce44    	fmla.4s	v4, v18, v16
10000d640: 4e217a92    	fcvtl2	v18.4s, v20.8h
10000d644: 4e31ce45    	fmla.4s	v5, v18, v17
10000d648: 0e217ab2    	fcvtl	v18.4s, v21.4h
10000d64c: 4e30ce46    	fmla.4s	v6, v18, v16
10000d650: 4e217ab0    	fcvtl2	v16.4s, v21.8h
10000d654: 4e31ce07    	fmla.4s	v7, v16, v17
10000d658: 910021ef    	add	x15, x15, #0x8
10000d65c: 91008231    	add	x17, x17, #0x20
10000d660: eb0b01ff    	cmp	x15, x11
10000d664: 54fffd43    	b.lo	0x10000d60c <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x460>
10000d668: 8b080948    	add	x8, x10, x8, lsl #2
10000d66c: 4e21d400    	fadd.4s	v0, v0, v1
10000d670: 4e23d441    	fadd.4s	v1, v2, v3
10000d674: 4e25d482    	fadd.4s	v2, v4, v5
10000d678: 4e27d4c3    	fadd.4s	v3, v6, v7
10000d67c: 6e20d400    	faddp.4s	v0, v0, v0
10000d680: 7e30d800    	faddp.2s	s0, v0
10000d684: 6e21d421    	faddp.4s	v1, v1, v1
10000d688: 7e30d821    	faddp.2s	s1, v1
10000d68c: 2d000500    	stp	s0, s1, [x8]
10000d690: 6e22d440    	faddp.4s	v0, v2, v2
10000d694: 7e30d800    	faddp.2s	s0, v0
10000d698: 6e23d461    	faddp.4s	v1, v3, v3
10000d69c: 7e30d821    	faddp.2s	s1, v1
10000d6a0: 2d010500    	stp	s0, s1, [x8, #0x8]
10000d6a4: 91001130    	add	x16, x9, #0x4
10000d6a8: 8b0e01ad    	add	x13, x13, x14
10000d6ac: aa0903e8    	mov	x8, x9
10000d6b0: eb02021f    	cmp	x16, x2
10000d6b4: 54fff949    	b.ls	0x10000d5dc <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x430>
10000d6b8: 17fffeca    	b	0x10000d1e0 <__ZZN8qwen_cpu12_GLOBAL__N_14Pool3runIZNS_5Model4Impl7projectERNS0_6TensorEPKfPfiRS1_EUlmmiE_EEvmT_ENUlPvmmiE_8__invokeESD_mmi+0x34>
