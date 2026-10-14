/* Combinadic unit tests: exact values, roundtrips, hostile ranks. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/combinadic.h"
#include "zpse/mask.h"
#include "zpse/status.h"

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL " __VA_ARGS__); printf("\n"); fails++; } \
} while (0)

static uint8_t comb_scratch[8192];

int main(void) {
    /* Exact binomials. */
    CHECK(zpse_comb_choose_u256(5, 2).w[0] == 10u, "C(5,2)==10");
    CHECK(zpse_comb_choose_u256(10, 0).w[0] == 1u, "C(10,0)==1");
    CHECK(zpse_u256_is_zero(zpse_comb_choose_u256(4, 5)), "C(4,5)==0");
    CHECK(zpse_comb_choose_u256(256, 2).w[0] == 32640u, "C(256,2)==32640");

    /* Payload lengths: C(5,2)=10 -> 1 B; C(256,128) is 252 bits -> 32 B. */
    {
        int wide = 0;
        CHECK(zpse_comb_payload_len(5, 2, &wide) == 1 && !wide, "L(5,2)==1");
        wide = 0;
        CHECK(zpse_comb_payload_len(256, 128, &wide) == 32 && wide, "L(256,128)==32 wide");
        wide = 0;
        CHECK(zpse_comb_payload_len(8, 3, &wide) == 1, "L(8,3)==1 (C=56)");
    }

    /* Known ranks: colex Rank = sum C(c_j, j), j = 1-based. */
    {
        uint16_t c1[] = {0, 1};       /* C(0,1)+C(1,2) = 0 */
        uint16_t c2[] = {1, 2};       /* 1 + 1 = 2 */
        uint16_t c3[] = {1, 2, 4};    /* 1 + 1 + 4 = 6 */
        CHECK(zpse_comb_rank(c1, 2).w[0] == 0u, "rank{0,1}==0");
        CHECK(zpse_comb_rank(c2, 2).w[0] == 2u, "rank{1,2}==2");
        CHECK(zpse_comb_rank(c3, 3).w[0] == 6u, "rank{1,2,4}==6");
    }

    /* Exhaustive 8-choose-3 roundtrip (56 subsets). */
    {
        uint16_t c[3]; int n = 0;
        uint8_t buf[32]; uint32_t w[8];
        for (c[0] = 0; c[0] < 8; c[0]++)
        for (c[1] = (uint16_t)(c[0] + 1); c[1] < 8; c[1]++)
        for (c[2] = (uint16_t)(c[1] + 1); c[2] < 8; c[2]++) {
            int bl = zpse_comb_rank_to_bytes(c, 3, 8, buf, sizeof buf);
            uint32_t w2[8] = {0,0,0,0,0,0,0,0};
            int rc;
            CHECK(bl == 1, "8c3 len==1");
            rc = zpse_comb_unrank(8, 3, buf, (uint16_t)bl, w2, comb_scratch);
            CHECK(rc == 0, "8c3 unrank ok");
            CHECK(zpse_mask_test(w2, c[0]) && zpse_mask_test(w2, c[1]) &&
                  zpse_mask_test(w2, c[2]), "8c3 bits set");
            CHECK(zpse_mask_count_bits(w2, 8) == 3, "8c3 popcount");
            (void)w; n++;
        }
        CHECK(n == 56, "8c3 count==56 (got %d)", n);
    }

    /* K=1 sweep over Nt=256: rank == position. */
    for (int pos = 0; pos < 256; pos++) {
        uint16_t c = (uint16_t)pos;
        uint8_t buf[32]; uint32_t w[8] = {0,0,0,0,0,0,0,0};
        int bl = zpse_comb_rank_to_bytes(&c, 1, 256, buf, sizeof buf);
        int rc;
        if (bl != 1) { CHECK(0, "K=1 len pos %d", pos); continue; }
        rc = zpse_comb_unrank(256, 1, buf, 1, w, comb_scratch);
        if (rc != 0 || !zpse_mask_test(w, (uint16_t)pos) ||
            zpse_mask_count_bits(w, 256) != 1) {
            CHECK(0, "K=1 roundtrip pos %d", pos);
        }
    }

    /* Hostile: rank == C(N,K) must fail; wrong length must fail. */
    {
        uint32_t w[8] = {0,0,0,0,0,0,0,0};
        uint8_t over[1] = {56};       /* C(8,3)=56, max valid 55 */
        uint8_t two[2] = {5, 0};      /* L(8,3)=1, overlong */
        CHECK(zpse_comb_unrank(8, 3, over, 1, w, comb_scratch) ==
              PSE_BAD_ENUMERATIVE_RANK, "rank==C rejected");
        CHECK(zpse_comb_unrank(8, 3, two, 2, w, comb_scratch) ==
              PSE_BAD_ENUMERATIVE_RANK, "overlong rank rejected");
        CHECK(zpse_comb_unrank(8, 0, over, 1, w, comb_scratch) ==
              PSE_BAD_ARG, "k=0 rejected");
    }

    /* Mode-5 mask integration: max rank C(N,K)-1 sets the top subset. */
    {
        /* Nt=8,K=3 top subset is {5,6,7}: rank = C(5,1)+C(6,2)+C(7,3)
         * = 5+15+35 = 55. */
        uint16_t c[] = {5, 6, 7};
        uint8_t buf[32]; uint32_t w[8] = {0,0,0,0,0,0,0,0};
        int bl = zpse_comb_rank_to_bytes(c, 3, 8, buf, sizeof buf);
        CHECK(bl == 1 && buf[0] == 55, "top rank==55");
        CHECK(zpse_mask_decode(5, buf, 1, 8, 3, w, comb_scratch) == 0,
              "mode5 decode ok");
        CHECK(zpse_mask_test(w, 5) && zpse_mask_test(w, 6) &&
              zpse_mask_test(w, 7), "mode5 top bits");
    }

    printf(fails ? "COMBINADIC: %d FAILURES\n" : "COMBINADIC: ALL OK\n", fails);
    return fails != 0;
}
